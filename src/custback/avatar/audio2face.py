"""NVIDIA Audio2Face-3D driver: audio-driven facial animation over gRPC.

Audio2Face-3D generates facial motion from speech audio; it does not render
pixels. This driver streams microphone (or WAV) audio to an Audio2Face-3D
gRPC endpoint — typically the Audio2Face-3D NIM container running next to a
suitable NVIDIA GPU — receives ARKit-style blendshape weights back, and
feeds them into the shared :class:`~custback.avatar.state.FaceState`
contract that the rigs render. Head sway comes from the idle animator, as
Audio2Face animates skin, jaw, tongue, and eyes but not head translation.

Requirements (the ``audio2face`` extra): NVIDIA's
``nvidia-audio2face-3d`` service bindings and their ``nvidia-ace`` shared
messages, ``grpcio``, and ``sounddevice`` for microphone capture only. WAV
streaming does not import or require the PortAudio binding. The service
endpoint is configured with ``driver.audio2face.url``; see the Audio2Face-3D
documentation for supported GPUs (data-center parts and GeForce RTX 3080
and up — the RTX 3060 is below the supported list, which is why the vision
driver is the local default).
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..api.security import (
    SecurityConfigurationError,
    create_client_ssl_context,
    validate_outbound_endpoint,
)
from .config import Audio2FaceConfig
from .drivers import (
    DriverUnavailableError,
    FaceDriver,
    ThreadSafeLatestState,
    idle_blink,
    idle_pose,
)
from .state import ARKIT_BLENDSHAPES, FaceState

log = logging.getLogger(__name__)

_CHANNEL_SET = frozenset(ARKIT_BLENDSHAPES)
_STALE_AFTER_S = 2.0
_RECONNECT_DELAY_S = 2.0
_CLOSE_TIMEOUT_S = 3.0
_GRPC_CHANNEL_OPTIONS = (("grpc.enable_http_proxy", 0),)
_LIFECYCLE_THREAD_FACTORY = threading.Thread


def arkit_channel_name(name: str) -> str | None:
    """Normalize an Audio2Face blendshape name to the shared channel set.

    Audio2Face-3D emits the ARKit set in PascalCase (``EyeBlinkLeft``);
    the renderers use MediaPipe's lowerCamelCase (``eyeBlinkLeft``).
    Unknown names (custom tongue/emotion channels) are ignored.
    """
    if not name:
        return None
    candidate = name[0].lower() + name[1:]
    return candidate if candidate in _CHANNEL_SET else None


def map_blendshape_frame(names: list[str], values: list[float]) -> dict[str, float]:
    """Convert one Audio2Face animation frame to renderer channels."""
    weights: dict[str, float] = {}
    for name, value in zip(names, values):
        channel = arkit_channel_name(name)
        if channel is not None:
            weights[channel] = min(1.0, max(0.0, float(value)))
    return weights


class AudioSource:
    """Produces 16-bit mono PCM chunks at the configured sample rate.

    Returning ``None`` marks a finite clip complete; an empty byte string
    means that a live source has no audio available yet.
    """

    def read(self, frames: int) -> bytes | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class WavAudioSource(AudioSource):
    """Loops a 16-bit mono WAV file in real time (useful for testing)."""

    def __init__(self, path: str | Path, sample_rate: int):
        self.path = Path(path).expanduser()
        self._wav = wave.open(str(self.path), "rb")
        if self._wav.getnchannels() != 1:
            raise DriverUnavailableError(f"{self.path.name} must be mono")
        if self._wav.getsampwidth() != 2:
            raise DriverUnavailableError(f"{self.path.name} must be 16-bit PCM")
        if self._wav.getframerate() != sample_rate:
            raise DriverUnavailableError(
                f"{self.path.name} is {self._wav.getframerate()} Hz; "
                f"the configured sample_rate is {sample_rate} Hz"
            )

    def read(self, frames: int) -> bytes:
        data = self._wav.readframes(frames)
        if len(data) < frames * 2:  # loop
            self._wav.rewind()
            data += self._wav.readframes(frames - len(data) // 2)
        return data

    def close(self) -> None:
        self._wav.close()


class MicrophoneSource(AudioSource):
    """Captures the default input device via the sounddevice binding."""

    def __init__(self, sample_rate: int):
        try:
            import sounddevice
        except (ImportError, OSError) as exc:  # OSError: missing PortAudio
            raise DriverUnavailableError(
                "sounddevice (and the PortAudio library) is required for "
                "microphone capture; install the [audio2face] extra"
            ) from exc
        self._stream = sounddevice.RawInputStream(
            samplerate=sample_rate, channels=1, dtype="int16"
        )
        self._stream.start()

    def read(self, frames: int) -> bytes:
        data, _overflowed = self._stream.read(frames)
        return bytes(data)

    def close(self) -> None:
        abort = getattr(self._stream, "abort", None)
        if callable(abort):
            abort()
        else:  # pragma: no cover - legacy sounddevice stream fallback
            self._stream.stop()
        self._stream.close()


def create_audio_source(cfg: Audio2FaceConfig) -> AudioSource:
    if cfg.audio_source == "microphone":
        return MicrophoneSource(cfg.sample_rate)
    return WavAudioSource(cfg.audio_source, cfg.sample_rate)


def microphone_available() -> bool:
    """Return whether the optional microphone binding can load PortAudio."""

    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


@dataclass(frozen=True)
class _Protocol:
    """Published Audio2Face-3D/ACE gRPC surface targeted by this driver."""

    grpc: Any
    stub_class: Any
    audio_stream: Any
    audio_stream_header: Any
    audio_header: Any
    audio_with_emotion: Any
    end_of_audio: Any


@dataclass
class _PendingInterruption:
    """One detached native lifecycle method with retained ownership."""

    kind: str
    resource: Any
    entered: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error: BaseException | None = None


@dataclass
class _OwnedAudio2FaceGeneration:
    """All native identities retained until their worker is terminal."""

    source: AudioSource | None
    call: Any | None
    channel: Any | None
    worker: threading.Thread | None
    interruptions: list[_PendingInterruption] = field(default_factory=list)
    completed_interruptions: set[tuple[str, int]] = field(default_factory=set)


def _load_protocol() -> _Protocol:
    try:
        import grpc
        from nvidia_ace.audio_pb2 import AudioHeader
        from nvidia_audio2face_3d.audio2face_pb2_grpc import (
            A2FControllerServiceStub,
        )
        from nvidia_audio2face_3d.messages_pb2 import (
            AudioWithEmotion,
            AudioWithEmotionStream,
            AudioWithEmotionStreamHeader,
        )
    except Exception as exc:
        raise DriverUnavailableError(
            "the audio2face driver needs compatible nvidia-audio2face-3d, "
            "nvidia-ace, protobuf, and grpcio bindings; install the "
            "[audio2face] extra and see NVIDIA's Audio2Face-3D NIM "
            "documentation"
        ) from exc
    return _Protocol(
        grpc=grpc,
        stub_class=A2FControllerServiceStub,
        audio_stream=AudioWithEmotionStream,
        audio_stream_header=AudioWithEmotionStreamHeader,
        audio_header=AudioHeader,
        audio_with_emotion=AudioWithEmotion,
        end_of_audio=AudioWithEmotionStream.EndOfAudio,
    )


def protocol_available() -> bool:
    """Probe every generated module and message required by the driver."""

    try:
        _load_protocol()
    except DriverUnavailableError:
        return False
    return True


class Audio2FaceDriver(FaceDriver):
    """Streams audio to Audio2Face-3D and renders its blendshape frames."""

    name = "audio2face"
    device = "grpc"

    def __init__(self, cfg: Audio2FaceConfig):
        self.cfg = cfg
        self._endpoint = validate_outbound_endpoint(
            cfg.url,
            kind="grpc",
            label="driver.audio2face.url",
            allow_empty=True,
            require_port=True,
        )
        self._root_certificates: bytes | None = None
        self._private_key: bytes | None = None
        self._certificate_chain: bytes | None = None
        if self._endpoint is not None:
            # Parse the CA and client keypair eagerly with OpenSSL, then retain
            # the exact bytes gRPC will use. Reconnects never reread mutable
            # trust/identity files and can never downgrade after TLS failure.
            create_client_ssl_context(
                self._endpoint,
                ca_file=cfg.tls_ca_file,
                certfile=cfg.tls_certfile,
                keyfile=cfg.tls_keyfile,
                label="driver.audio2face",
            )
            try:
                if cfg.tls_ca_file:
                    self._root_certificates = (
                        Path(cfg.tls_ca_file).expanduser().read_bytes()
                    )
                if cfg.tls_keyfile:
                    self._private_key = Path(cfg.tls_keyfile).expanduser().read_bytes()
                    self._certificate_chain = (
                        Path(cfg.tls_certfile).expanduser().read_bytes()
                    )
            except OSError as exc:
                raise SecurityConfigurationError(
                    "driver.audio2face TLS material changed or became unavailable "
                    "during startup"
                ) from exc
        self._state = ThreadSafeLatestState()
        self._stop = threading.Event()
        self._operation_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._source: AudioSource | None = None
        self._call: Any | None = None
        self._channel: Any | None = None
        self._survivors: list[tuple[str, Any, BaseException]] = []
        self._interruptions: list[_PendingInterruption] = []
        self._owned_generation: _OwnedAudio2FaceGeneration | None = None
        self._closed = False
        self._last_error: str = ""

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        """Verify dependencies and audio input, then start streaming."""

        with self._operation_lock:
            self._start_owned()

    def _start_owned(self) -> None:
        if not self.cfg.url:
            raise DriverUnavailableError("driver.audio2face.url is not configured")
        with self._lifecycle_lock:
            if self._closed or self._stop.is_set() or self._survivors:
                raise DriverUnavailableError("audio2face driver is already closed")
            if self._worker is not None:
                raise DriverUnavailableError("audio2face driver is already started")
        protocol = _load_protocol()
        source = create_audio_source(self.cfg)
        rejected = ""
        worker: threading.Thread | None = None
        try:
            worker = threading.Thread(
                target=self._stream_forever,
                args=(protocol, source),
                name="custback-audio2face",
                daemon=True,
            )
            with self._lifecycle_lock:
                if self._closed or self._stop.is_set() or self._survivors:
                    rejected = "audio2face driver is already closed"
                elif self._worker is not None:
                    rejected = "audio2face driver is already started"
                else:
                    self._source = source
                    self._worker = worker
                    # Publish only a successfully started worker. Holding the
                    # lock closes the small join-before-start race with close().
                    worker.start()
        except BaseException:
            with self._lifecycle_lock:
                if self._source is source:
                    self._source = None
                if worker is not None and self._worker is worker:
                    self._worker = None
            try:
                source.close()
            except BaseException as exc:
                log.debug("audio2face source close failed", exc_info=True)
                self._remember_survivor("source", source, exc)
            raise
        if rejected:
            try:
                source.close()
            except BaseException as exc:
                log.debug("audio2face source close failed", exc_info=True)
                self._remember_survivor("source", source, exc)
            raise DriverUnavailableError(rejected)

    def close(self) -> None:
        """Interrupt the active stream and synchronously reclaim its worker."""

        deadline = time.monotonic() + _CLOSE_TIMEOUT_S
        # Publish shutdown intent without waiting for a concurrent start/close.
        # A start that is provisioning a source will reject it at publication.
        self._stop.set()
        with self._lifecycle_lock:
            self._closed = True
        if not self._operation_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            raise DriverUnavailableError(
                "audio2face lifecycle operation did not stop within "
                f"{_CLOSE_TIMEOUT_S:.1f}s"
            )
        try:
            self._close_owned(deadline)
        finally:
            self._operation_lock.release()

    def _close_owned(self, deadline: float) -> None:
        with self._lifecycle_lock:
            generation = self._owned_generation
            if generation is None:
                # Move every active identity into one terminal-ownership
                # generation. The individual fields are cleared so worker
                # finally blocks cannot double-close resources while close()
                # interrupts them, but the driver retains the complete set
                # until the exact worker is joined.
                generation = _OwnedAudio2FaceGeneration(
                    source=self._source,
                    call=self._call,
                    channel=self._channel,
                    worker=self._worker,
                )
                self._owned_generation = generation
                self._source = None
                self._call = None
                self._channel = None
            worker = generation.worker
            survivors, self._survivors = self._survivors, []
            pending, self._interruptions = self._interruptions, []

        # Closing the capture first wakes a request generator blocked in read().
        # Cancelling the RPC then wakes a response iterator blocked in next().
        owned = [
            ("source", generation.source),
            ("call", generation.call),
            ("channel", generation.channel),
        ]
        owned.extend((kind, resource) for kind, resource, _error in survivors)
        retry: list[tuple[str, Any]] = []
        still_pending: list[_PendingInterruption] = []
        for interruption in pending:
            if interruption.done.is_set():
                if interruption.error is not None:
                    retry.append((interruption.kind, interruption.resource))
                else:
                    generation.completed_interruptions.add(
                        self._interruption_key(interruption.kind, interruption.resource)
                    )
            else:
                # A completion racing this observation remains conservatively
                # owned and is classified on the next settle/retry pass.
                still_pending.append(interruption)
        pending = still_pending
        busy = {self._interruption_key(item.kind, item.resource) for item in pending}
        candidates = [
            (kind, resource)
            for kind, resource in owned
            if resource is not None
            and self._interruption_key(kind, resource)
            not in generation.completed_interruptions
            and self._interruption_key(kind, resource) not in busy
        ]
        candidates.extend(retry)
        pending.extend(self._begin_interruptions(candidates, deadline))
        self._retain_generation_interruptions(generation, pending)
        failures, active = self._settle_interruptions(pending, deadline)
        self._record_completed_interruptions(generation, pending, failures, active)
        self._retain_interruptions(active)

        if worker is threading.current_thread():
            self._remember_survivors(failures)
            raise DriverUnavailableError(
                "cannot close audio2face from its streaming worker"
            )
        if worker is not None:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                self._remember_survivors(failures)
                raise DriverUnavailableError(
                    f"audio2face worker did not stop within {_CLOSE_TIMEOUT_S:.1f}s"
                )

        # Registration/release can race the first ownership snapshot. Once the
        # worker is terminal no more resources can appear, so retry every late
        # survivor before close is allowed to report success.
        with self._lifecycle_lock:
            late, self._survivors = self._survivors, []
        late_pending = self._begin_interruptions(
            [(kind, resource) for kind, resource, _error in late],
            deadline,
        )
        self._retain_generation_interruptions(generation, late_pending)
        late_failures, late_active = self._settle_interruptions(late_pending, deadline)
        failures.extend(late_failures)
        active.extend(late_active)
        self._record_completed_interruptions(
            generation, late_pending, late_failures, late_active
        )
        self._retain_interruptions(late_active)
        if active:
            self._remember_survivors(failures)
            raise DriverUnavailableError(
                "audio2face native interruption did not stop within "
                f"{_CLOSE_TIMEOUT_S:.1f}s"
            )
        if failures:
            self._remember_survivors(failures)
            names = ", ".join(
                dict.fromkeys(kind for kind, _resource, _error in failures)
            )
            raise DriverUnavailableError(
                f"audio2face lifecycle interruption failed for: {names}"
            )
        if time.monotonic() > deadline:
            raise DriverUnavailableError(
                "audio2face close exceeded its "
                f"{_CLOSE_TIMEOUT_S:.1f}s shutdown deadline"
            )
        with self._lifecycle_lock:
            if self._worker is worker:
                self._worker = None
            if self._owned_generation is generation:
                self._owned_generation = None

    @staticmethod
    def _interruption_key(kind: str, resource: Any) -> tuple[str, int]:
        return kind, id(resource)

    @staticmethod
    def _retain_generation_interruptions(
        generation: _OwnedAudio2FaceGeneration,
        pending: list[_PendingInterruption],
    ) -> None:
        for interruption in pending:
            if not any(interruption is saved for saved in generation.interruptions):
                generation.interruptions.append(interruption)

    @classmethod
    def _record_completed_interruptions(
        cls,
        generation: _OwnedAudio2FaceGeneration,
        pending: list[_PendingInterruption],
        failures: list[tuple[str, Any, BaseException]],
        active: list[_PendingInterruption],
    ) -> None:
        """Remember successful attempts without racing terminal ownership."""

        unresolved = {
            cls._interruption_key(kind, resource) for kind, resource, _error in failures
        }
        unresolved.update(
            cls._interruption_key(item.kind, item.resource) for item in active
        )
        for interruption in pending:
            key = cls._interruption_key(interruption.kind, interruption.resource)
            if key not in unresolved:
                generation.completed_interruptions.add(key)

    def _begin_interruptions(
        self,
        owned: list[tuple[str, Any | None]],
        deadline: float,
    ) -> list[_PendingInterruption]:
        """Initiate detached native interruption in lifecycle order."""

        pending: list[_PendingInterruption] = []
        seen: set[int] = set()
        for expected_kind in ("source", "call", "channel"):
            for kind, resource in owned:
                if kind != expected_kind or resource is None or id(resource) in seen:
                    continue
                seen.add(id(resource))

                interruption = _PendingInterruption(kind, resource)

                def interrupt(item: _PendingInterruption = interruption) -> None:
                    item.entered.set()
                    try:
                        if item.kind == "call":
                            cancel = getattr(item.resource, "cancel", None)
                            if callable(cancel):
                                cancel()
                        else:
                            item.resource.close()
                    except BaseException as exc:
                        item.error = exc
                        log.debug(
                            "audio2face %s interruption failed",
                            item.kind,
                            exc_info=True,
                        )
                    finally:
                        item.done.set()

                try:
                    interruption.thread = _LIFECYCLE_THREAD_FACTORY(
                        target=interrupt,
                        name=f"custback-audio2face-close-{kind}",
                        daemon=True,
                    )
                    interruption.thread.start()
                    interruption.entered.wait(max(0.0, deadline - time.monotonic()))
                except BaseException as exc:
                    interruption.error = exc
                    interruption.done.set()
                pending.append(interruption)
        return pending

    @staticmethod
    def _settle_interruptions(
        pending: list[_PendingInterruption], deadline: float
    ) -> tuple[
        list[tuple[str, Any, BaseException]],
        list[_PendingInterruption],
    ]:
        for interruption in pending:
            if interruption.done.is_set():
                continue
            interruption.done.wait(max(0.0, deadline - time.monotonic()))
        failures: list[tuple[str, Any, BaseException]] = []
        active: list[_PendingInterruption] = []
        for interruption in pending:
            if interruption.done.is_set():
                if interruption.error is not None:
                    failures.append(
                        (
                            interruption.kind,
                            interruption.resource,
                            interruption.error,
                        )
                    )
            else:
                # Never perform a second state read: completion may race this
                # branch, but retaining a now-terminal item is safe and retryable.
                active.append(interruption)
        return failures, active

    def _retain_interruptions(self, pending: list[_PendingInterruption]) -> None:
        if not pending:
            return
        with self._lifecycle_lock:
            for interruption in pending:
                if not any(interruption is saved for saved in self._interruptions):
                    self._interruptions.append(interruption)

    def _remember_survivor(
        self, kind: str, resource: Any, error: BaseException
    ) -> None:
        with self._lifecycle_lock:
            if not any(
                kind == saved_kind and resource is saved_resource
                for saved_kind, saved_resource, _saved_error in self._survivors
            ):
                self._survivors.append((kind, resource, error))
        # A failed native teardown is terminal: never reconnect alongside it.
        self._stop.set()

    def _remember_survivors(
        self, failures: list[tuple[str, Any, BaseException]]
    ) -> None:
        for kind, resource, error in failures:
            self._remember_survivor(kind, resource, error)

    def _register_channel(self, channel: Any) -> bool:
        """Publish a channel unless shutdown already owns the lifecycle."""

        with self._lifecycle_lock:
            if self._closed or self._stop.is_set() or self._channel is not None:
                return False
            self._channel = channel
            return True

    def _release_channel(self, channel: Any) -> None:
        with self._lifecycle_lock:
            if self._channel is not channel:
                return
            self._channel = None
        try:
            channel.close()
        except BaseException as exc:
            log.debug("audio2face channel close failed", exc_info=True)
            self._remember_survivor("channel", channel, exc)

    def _register_call(self, call: Any, channel: Any) -> bool:
        """Publish an RPC only while its channel still belongs to this session."""

        with self._lifecycle_lock:
            if (
                self._closed
                or self._stop.is_set()
                or self._channel is not channel
                or self._call is not None
            ):
                return False
            self._call = call
            return True

    def _release_call(self, call: Any) -> None:
        with self._lifecycle_lock:
            if self._call is call:
                self._call = None

    def _release_source(self, source: AudioSource) -> None:
        with self._lifecycle_lock:
            if self._source is not source:
                return
            self._source = None
        try:
            source.close()
        except BaseException as exc:
            log.debug("audio2face source close failed", exc_info=True)
            self._remember_survivor("source", source, exc)

    @property
    def healthy(self) -> bool:
        _weights, updated_at = self._state.snapshot()
        return (
            updated_at is not None and (time.monotonic() - updated_at) <= _STALE_AFTER_S
        )

    @property
    def last_error(self) -> str:
        return self._last_error

    # -- gRPC session ------------------------------------------------------
    def _stream_forever(self, protocol: _Protocol, source: AudioSource) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._run_session(protocol, source)
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    # Keep credential- and address-free diagnostics only.
                    self._last_error = type(exc).__name__
                    log.warning(
                        "audio2face stream interrupted (%s); reconnecting",
                        type(exc).__name__,
                    )
                if not self._stop.is_set():
                    self._stop.wait(_RECONNECT_DELAY_S)
        finally:
            self._release_source(source)

    def _requests(self, protocol: _Protocol, source: AudioSource) -> Iterator[Any]:
        if self._stop.is_set():
            return
        header = protocol.audio_stream(
            audio_stream_header=protocol.audio_stream_header(
                audio_header=protocol.audio_header(
                    audio_format=protocol.audio_header.AUDIO_FORMAT_PCM,
                    channel_count=1,
                    bits_per_sample=16,
                    samples_per_second=self.cfg.sample_rate,
                )
            )
        )
        yield header
        chunk_frames = max(1, self.cfg.sample_rate * self.cfg.chunk_ms // 1000)
        chunk_seconds = chunk_frames / self.cfg.sample_rate
        while not self._stop.is_set():
            started = time.monotonic()
            data = source.read(chunk_frames)
            if self._stop.is_set():
                return
            if data is None:
                yield protocol.audio_stream(end_of_audio=protocol.end_of_audio())
                return
            if not data:
                self._stop.wait(chunk_seconds)
                continue
            yield protocol.audio_stream(
                audio_with_emotion=protocol.audio_with_emotion(audio_buffer=data)
            )
            # WAV sources return immediately; pace them to real time so the
            # service animates at speech speed instead of draining the file.
            elapsed = time.monotonic() - started
            if elapsed < chunk_seconds:
                self._stop.wait(chunk_seconds - elapsed)

    def _run_session(self, protocol: _Protocol, source: AudioSource) -> None:
        endpoint = self._endpoint
        if endpoint is None:
            raise DriverUnavailableError("driver.audio2face.url is not configured")
        if endpoint.secure:
            credentials = protocol.grpc.ssl_channel_credentials(
                root_certificates=self._root_certificates,
                private_key=self._private_key,
                certificate_chain=self._certificate_chain,
            )
            channel = protocol.grpc.secure_channel(
                endpoint.authority,
                credentials,
                options=_GRPC_CHANNEL_OPTIONS,
            )
        else:
            channel = protocol.grpc.insecure_channel(
                endpoint.authority,
                options=_GRPC_CHANNEL_OPTIONS,
            )
        if not self._register_channel(channel):
            try:
                channel.close()
            except BaseException as exc:
                log.debug("audio2face channel close failed", exc_info=True)
                self._remember_survivor("channel", channel, exc)
            return
        try:
            stub = protocol.stub_class(channel)
            names: list[str] = []
            call = stub.ProcessAudioStream(self._requests(protocol, source))
            if not self._register_call(call, channel):
                cancel = getattr(call, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except BaseException as exc:
                        log.debug("audio2face RPC cancellation failed", exc_info=True)
                        self._remember_survivor("call", call, exc)
                return
            try:
                for message in call:
                    if self._stop.is_set():
                        return
                    header = getattr(message, "animation_data_stream_header", None)
                    if header is not None and header.ByteSize():
                        names = list(header.skel_animation_header.blend_shapes)
                        continue
                    animation = getattr(message, "animation_data", None)
                    if animation is None or not animation.ByteSize():
                        continue
                    for frame in animation.skel_animation.blend_shape_weights:
                        weights = map_blendshape_frame(names, list(frame.values))
                        if weights:
                            self._state.update_blendshapes(weights, time.monotonic())
            finally:
                self._release_call(call)
        finally:
            self._release_channel(channel)

    # -- renderer side ---------------------------------------------------
    def update(self, frame_bgr: np.ndarray | None, timestamp: float) -> FaceState:
        weights, updated_at = self._state.snapshot()
        yaw, pitch, roll = idle_pose(timestamp)
        state = FaceState(
            present=True, yaw=yaw, pitch=pitch, roll=roll, timestamp=timestamp
        )
        stale = updated_at is None or (time.monotonic() - updated_at) > _STALE_AFTER_S
        if not stale:
            for name, value in weights.items():
                state.set_channel(name, value)
        blink = idle_blink(timestamp)
        if blink > state.channel("eyeBlinkLeft"):
            state.set_channel("eyeBlinkLeft", blink)
            state.set_channel("eyeBlinkRight", blink)
        return state
