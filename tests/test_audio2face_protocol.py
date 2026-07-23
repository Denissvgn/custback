"""Contract tests against NVIDIA's published Audio2Face protobuf wheels."""

from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor

import pytest

if importlib.util.find_spec("nvidia_audio2face_3d") is None:
    pytest.skip(
        "the optional nvidia-audio2face-3d protocol package is not installed",
        allow_module_level=True,
    )

import grpc  # pyright: ignore[reportMissingModuleSource]
from nvidia_ace import animation_pb2  # pyright: ignore[reportMissingImports]
from nvidia_audio2face_3d import (  # pyright: ignore[reportMissingImports]
    audio2face_pb2_grpc,
    messages_pb2,
)

from custback.avatar.audio2face import (
    AudioSource,
    Audio2FaceDriver,
    _load_protocol,
    protocol_available,
)
from custback.avatar.config import Audio2FaceConfig


def _request_messages():
    protocol = _load_protocol()
    header = protocol.audio_stream(
        audio_stream_header=protocol.audio_stream_header(
            audio_header=protocol.audio_header(
                audio_format=protocol.audio_header.AUDIO_FORMAT_PCM,
                channel_count=1,
                samples_per_second=16_000,
                bits_per_sample=16,
            )
        )
    )
    chunk = protocol.audio_stream(
        audio_with_emotion=protocol.audio_with_emotion(audio_buffer=b"\x01\x00")
    )
    end = protocol.audio_stream(end_of_audio=protocol.end_of_audio())
    return protocol, (header, chunk, end)


def test_published_audio2face_messages_serialize_and_parse():
    assert protocol_available()
    protocol, requests = _request_messages()

    parsed_requests = [
        protocol.audio_stream.FromString(message.SerializeToString())
        for message in requests
    ]
    assert [message.WhichOneof("stream_part") for message in parsed_requests] == [
        "audio_stream_header",
        "audio_with_emotion",
        "end_of_audio",
    ]
    request_header = parsed_requests[0].audio_stream_header.audio_header
    assert request_header.samples_per_second == 16_000
    assert parsed_requests[1].audio_with_emotion.audio_buffer == b"\x01\x00"

    response = messages_pb2.A2F3DAnimationDataStream(
        animation_data=animation_pb2.AnimationData(
            skel_animation=animation_pb2.SkelAnimation(
                blend_shape_weights=[
                    animation_pb2.FloatArrayWithTimeCode(
                        time_code=0.0,
                        values=[0.75],
                    )
                ]
            )
        )
    )
    parsed_response = messages_pb2.A2F3DAnimationDataStream.FromString(
        response.SerializeToString()
    )
    assert list(
        parsed_response.animation_data.skel_animation.blend_shape_weights[0].values
    ) == pytest.approx([0.75])


def test_driver_uses_generated_in_process_audio2face_service():
    captured = []

    class Service(audio2face_pb2_grpc.A2FControllerServiceServicer):
        def ProcessAudioStream(self, request_iterator, _context):
            captured.extend(request_iterator)
            yield messages_pb2.A2F3DAnimationDataStream(
                animation_data_stream_header=(
                    messages_pb2.A2F3DAnimationDataStreamHeader(
                        skel_animation_header=animation_pb2.SkelAnimationHeader(
                            blend_shapes=["JawOpen", "EyeBlinkLeft"]
                        )
                    )
                )
            )
            yield messages_pb2.A2F3DAnimationDataStream(
                animation_data=animation_pb2.AnimationData(
                    skel_animation=animation_pb2.SkelAnimation(
                        blend_shape_weights=[
                            animation_pb2.FloatArrayWithTimeCode(
                                time_code=0.0,
                                values=[0.65, 0.25],
                            )
                        ]
                    )
                )
            )

    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    audio2face_pb2_grpc.add_A2FControllerServiceServicer_to_server(Service(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    server.start()

    class FiniteSource(AudioSource):
        def __init__(self):
            self.reads = 0

        def read(self, frames: int) -> bytes | None:
            self.reads += 1
            if self.reads == 1:
                return b"\x00\x00" * min(frames, 16)
            return None

    driver = Audio2FaceDriver(
        Audio2FaceConfig(
            url=f"grpc://127.0.0.1:{port}",
            chunk_ms=10,
        )
    )
    try:
        driver._run_session(_load_protocol(), FiniteSource())
    finally:
        server.stop(0).wait(5)

    assert [message.WhichOneof("stream_part") for message in captured] == [
        "audio_stream_header",
        "audio_with_emotion",
        "end_of_audio",
    ]
    assert captured[0].audio_stream_header.audio_header.samples_per_second == 16_000
    weights, updated_at = driver._state.snapshot()
    assert updated_at is not None
    assert weights == pytest.approx({"jawOpen": 0.65, "eyeBlinkLeft": 0.25})
