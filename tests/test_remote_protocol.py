"""Strict remote-renderer epoch envelopes and exact Hub provenance."""

from __future__ import annotations

import struct
import threading

import numpy as np
import pytest

from custback.hub import FrameHub
from custback.remote_protocol import (
    REMOTE_FRAME_HEADER_BYTES,
    RemoteFrameProtocolError,
    decode_remote_frame,
    encode_remote_frame,
)


def test_remote_frame_envelope_round_trips_kind_epoch_and_payload() -> None:
    jpeg = b"\xff\xd8bounded-jpeg\xff\xd9"
    raw = encode_remote_frame("raw-input", 41, jpeg, max_message_bytes=1024)
    rendered = encode_remote_frame(
        "rendered-output",
        41,
        jpeg,
        max_message_bytes=1024,
    )

    assert len(raw) == REMOTE_FRAME_HEADER_BYTES + len(jpeg)
    assert (
        decode_remote_frame(
            raw,
            expected_kind="raw-input",
            max_message_bytes=1024,
        ).raw_epoch
        == 41
    )
    decoded = decode_remote_frame(
        rendered,
        expected_kind="rendered-output",
        max_message_bytes=1024,
    )
    assert decoded.kind == "rendered-output"
    assert decoded.raw_epoch == 41
    assert decoded.jpeg == jpeg

    inactive = decode_remote_frame(
        encode_remote_frame("raw-input", 0, jpeg, max_message_bytes=1024),
        expected_kind="raw-input",
        max_message_bytes=1024,
    )
    assert inactive.raw_epoch == 0
    with pytest.raises(ValueError, match="positive raw epoch"):
        encode_remote_frame("rendered-output", 0, jpeg)


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda data: data.__setitem__(0, ord("X")), "unsupported"),
        (lambda data: data.__setitem__(8, 2), "unsupported"),
        (lambda data: data.__setitem__(9, 1), "kind"),
        (lambda data: data.__setitem__(10, 1), "reserved"),
        (
            lambda data: data.__setitem__(
                slice(20, 24),
                struct.pack(">I", len(data)),
            ),
            "length",
        ),
    ),
)
def test_remote_frame_envelope_rejects_header_tampering(mutate, match: str) -> None:
    message = bytearray(
        encode_remote_frame("rendered-output", 9, b"jpeg", max_message_bytes=1024)
    )
    mutate(message)
    with pytest.raises(RemoteFrameProtocolError, match=match):
        decode_remote_frame(
            bytes(message),
            expected_kind="rendered-output",
            max_message_bytes=1024,
        )


def test_remote_frame_envelope_enforces_exact_and_bounded_lengths() -> None:
    message = encode_remote_frame(
        "rendered-output",
        7,
        b"jpeg",
        max_message_bytes=REMOTE_FRAME_HEADER_BYTES + 4,
    )
    for invalid in (message[:-1], message + b"x"):
        with pytest.raises(RemoteFrameProtocolError, match="length"):
            decode_remote_frame(
                invalid,
                expected_kind="rendered-output",
                max_message_bytes=1024,
            )
    with pytest.raises(ValueError, match="exceeds"):
        encode_remote_frame(
            "rendered-output",
            7,
            b"jpeg",
            max_message_bytes=REMOTE_FRAME_HEADER_BYTES + 3,
        )
    with pytest.raises(RemoteFrameProtocolError, match="length"):
        decode_remote_frame(
            b"legacy-jpeg",
            expected_kind="rendered-output",
            max_message_bytes=1024,
        )


def test_hub_binds_slot_sequences_and_requires_explicit_current_epoch() -> None:
    hub = FrameHub()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    hub.publish_raw(frame)
    _local, local_sequence = hub.raw.get(-1, 0.0)
    assert hub.remote_raw_epoch_for_sequence(local_sequence) == 0

    session = hub.remote_client_connected()
    assert hub.remote_raw_epoch_for_sequence(local_sequence) is None
    with pytest.raises(ValueError, match="positive integer"):
        hub.push_remote_frame(frame, raw_epoch=0, session_id=session)
    hub.publish_remote_raw(frame, 11)
    _remote, remote_sequence = hub.raw.get(local_sequence, 0.0)
    assert hub.remote_raw_epoch_for_sequence(remote_sequence) == 11

    assert not hub.push_remote_frame(
        frame,
        raw_epoch=10,
        session_id=session,
    )
    assert hub.remote_in.latest()[0] is None
    assert hub.push_remote_frame(
        frame,
        raw_epoch=11,
        session_id=session,
    )
    proof = hub.remote_frame_proof_status(1.0)
    assert proof[0] is frame
    assert proof[2:4] == (11, session)
    assert not hub.push_remote_frame(
        frame.copy(),
        raw_epoch=11,
        session_id=session,
    )
    assert hub.remote_frame_proof_status(1.0)[4] == proof[4]

    hub.publish_remote_raw(frame, 12)
    assert not hub.push_remote_frame(
        frame,
        raw_epoch=11,
        session_id=session,
    )
    with pytest.raises(TypeError):
        hub.push_remote_frame(frame, session)  # type: ignore[misc]
    hub.remote_client_disconnected(session)
    assert hub.remote_raw_epoch_for_sequence(remote_sequence) is None


def test_remote_raw_exposure_and_replay_admission_are_session_atomic() -> None:
    hub = FrameHub()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    session = hub.remote_client_connected()
    admission_started = threading.Event()
    release_admission = threading.Event()
    disconnect_started = threading.Event()
    disconnected = threading.Event()
    publication: list[bool] = []

    def admit(observed_session: int) -> bool:
        assert observed_session == session
        admission_started.set()
        assert release_admission.wait(1.0)
        return True

    publishing = threading.Thread(
        target=lambda: publication.append(hub.publish_remote_raw(frame, 1, admit=admit))
    )
    publishing.start()
    assert admission_started.wait(1.0)

    def disconnect() -> None:
        disconnect_started.set()
        hub.remote_client_disconnected(session)
        disconnected.set()

    disconnecting = threading.Thread(target=disconnect)
    disconnecting.start()
    assert disconnect_started.wait(1.0)
    assert not disconnected.wait(0.05)

    release_admission.set()
    publishing.join(1.0)
    disconnecting.join(1.0)
    assert not publishing.is_alive()
    assert not disconnecting.is_alive()
    assert publication == [True]
    assert disconnected.is_set()
    assert hub.active_remote_session() is None
    assert hub.remote_in.latest()[0] is None
    _published, sequence = hub.raw.get(-1, 0.0)
    assert hub.remote_raw_epoch_for_sequence(sequence) is None


def test_hub_raw_sequence_epoch_binding_is_bounded() -> None:
    hub = FrameHub()
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    session = hub.remote_client_connected()
    sequences: list[int] = []

    for raw_epoch in range(1, 2051):
        hub.publish_remote_raw(frame, raw_epoch)
        _published, sequence = hub.raw.get(-1, 0.0)
        sequences.append(sequence)

    assert hub.remote_raw_epoch_for_sequence(sequences[0]) is None
    assert hub.remote_raw_epoch_for_sequence(sequences[1]) is None
    assert hub.remote_raw_epoch_for_sequence(sequences[2]) == 3
    assert hub.remote_raw_epoch_for_sequence(sequences[-1]) == 2050
    assert hub.push_remote_frame(
        frame,
        raw_epoch=2050,
        session_id=session,
    )


def test_canvas_change_discards_old_slot_epoch_bindings() -> None:
    hub = FrameHub()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    hub.configure_canvas((6, 4))
    hub.publish_remote_raw(frame, 1)
    _published, sequence = hub.raw.get(-1, 0.0)
    assert hub.remote_raw_epoch_for_sequence(sequence) == 1

    hub.configure_canvas((8, 4))
    assert hub.remote_raw_epoch_for_sequence(sequence) is None
