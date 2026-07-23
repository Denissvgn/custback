#!/usr/bin/env python3
"""In-network probes for the packaged two-host release system test."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import socket
import ssl
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import websockets


def _token(path: str) -> str:
    value = Path(path).read_text().strip()
    if not value:
        raise RuntimeError("token file is empty")
    return value


def _context(ca_file: str, *, insecure: bool = False) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()  # noqa: SLF001 - observer only
    context = ssl.create_default_context(cafile=ca_file)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def _code(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    return detail.get("code") if isinstance(detail, dict) else None


def _websocket_headers(headers: dict[str, str]) -> dict[str, Any]:
    """Return the version-dependent header argument for ``websockets.connect``."""
    parameters = inspect.signature(websockets.connect).parameters
    name = (
        "additional_headers" if "additional_headers" in parameters else "extra_headers"
    )
    return {name: headers}


def _request(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {_token(args.token_file)}"}
    verify: str | bool = False if args.insecure else args.ca_file
    with httpx.Client(verify=verify, timeout=args.timeout) as client:
        response = client.request(args.method, args.url, headers=headers)
    try:
        body: Any = response.json()
    except ValueError:
        body = {"text": response.text[:200]}
    observed_code = _code(body)
    if args.expect_status is not None and response.status_code != args.expect_status:
        raise RuntimeError(
            f"HTTP status {response.status_code}, expected {args.expect_status}"
        )
    if args.expect_code is not None and observed_code != args.expect_code:
        raise RuntimeError(
            f"error code {observed_code!r}, expected {args.expect_code!r}"
        )
    return {
        "status": response.status_code,
        "code": observed_code,
        "body": body,
    }


def command_http(args: argparse.Namespace) -> int:
    print(json.dumps(_request(args), sort_keys=True))
    return 0


def command_wait_http(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.wait_timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            result = _request(args)
            print(json.dumps(result, sort_keys=True))
            return 0
        except Exception as exc:  # readiness deliberately retries every failure
            last_error = type(exc).__name__
            time.sleep(0.2)
    raise RuntimeError(f"endpoint did not become ready ({last_error})")


def command_browser_session_create(args: argparse.Namespace) -> int:
    verify: str | bool = False if args.insecure else args.ca_file
    headers = {"Origin": args.origin}
    with httpx.Client(verify=verify, timeout=args.timeout) as client:
        response = client.post(
            args.url,
            headers=headers,
            json={"token": _token(args.token_file)},
        )
    if response.status_code != 204:
        raise RuntimeError(f"browser session creation returned {response.status_code}")
    cookie = response.cookies.get("custback_session")
    if not cookie:
        raise RuntimeError("browser session response omitted its secure cookie")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(args.cookie_file, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = cookie.encode("ascii")
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short browser cookie write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"status": response.status_code, "session_created": True}))
    return 0


def command_browser_session_check(args: argparse.Namespace) -> int:
    cookie = Path(args.cookie_file).read_text().strip()
    if not cookie:
        raise RuntimeError("browser session cookie file is empty")
    verify: str | bool = False if args.insecure else args.ca_file
    with httpx.Client(verify=verify, timeout=args.timeout) as client:
        response = client.get(
            args.url,
            headers={"Origin": args.origin},
            cookies={"custback_session": cookie},
        )
    if response.status_code != 200:
        raise RuntimeError(f"existing browser session returned {response.status_code}")
    print(json.dumps({"status": response.status_code, "session_valid": True}))
    return 0


def _privacy_slate(height: int, width: int) -> np.ndarray:
    tile = max(2, min(height, width) // 8)
    yy, xx = np.indices((height, width), dtype=np.int32)
    checker = ((yy // tile) + (xx // tile)) & 1
    slate = np.empty((height, width, 3), dtype=np.uint8)
    slate[checker == 0] = (24, 27, 32)
    slate[checker == 1] = (36, 40, 48)
    return slate


def _observe_frame(payload: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise RuntimeError("output WebSocket returned a non-binary message")
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError("output WebSocket returned an invalid JPEG")
    expected = _privacy_slate(frame.shape[0], frame.shape[1])
    error = float(np.mean(np.abs(frame.astype(np.int16) - expected.astype(np.int16))))
    slate = error <= args.slate_max_error
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "height": int(frame.shape[0]),
        "width": int(frame.shape[1]),
        "privacy_slate": slate,
        "slate_error": round(error, 3),
    }


def _assert_expectation(observations: list[dict[str, Any]], expectation: str) -> None:
    slate_count = sum(bool(item["privacy_slate"]) for item in observations)
    if expectation == "slate" and slate_count != len(observations):
        raise RuntimeError(
            "recorded output was not the fixed privacy slate throughout "
            f"({slate_count}/{len(observations)} frames)"
        )
    if expectation == "rendered" and slate_count:
        raise RuntimeError(
            f"nominal renderer output contained {slate_count} privacy-slate frame(s)"
        )


async def _output_frame(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {_token(args.token_file)}"}

    async def receive_frame() -> dict[str, Any]:
        async with websockets.connect(
            args.url,
            ssl=_context(args.ca_file, insecure=args.insecure),
            origin=args.origin,
            max_size=args.max_bytes,
            **_websocket_headers(headers),
        ) as websocket:
            return _observe_frame(await websocket.recv(), args)

    observation = await asyncio.wait_for(receive_frame(), timeout=args.timeout)
    _assert_expectation([observation], args.expect)
    return observation


def command_output_frame(args: argparse.Namespace) -> int:
    print(json.dumps(asyncio.run(_output_frame(args)), sort_keys=True))
    return 0


async def _record_output(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {_token(args.token_file)}"}
    observations: list[dict[str, Any]] = []

    async def receive_frames() -> None:
        async with websockets.connect(
            args.url,
            ssl=_context(args.ca_file, insecure=args.insecure),
            origin=args.origin,
            max_size=args.max_bytes,
            **_websocket_headers(headers),
        ) as websocket:
            for _index in range(args.frames):
                observations.append(_observe_frame(await websocket.recv(), args))

    await asyncio.wait_for(receive_frames(), timeout=args.timeout)
    _assert_expectation(observations, args.expect)
    return {
        "frames": len(observations),
        "privacy_slate_frames": sum(
            bool(item["privacy_slate"]) for item in observations
        ),
        "observations": observations,
    }


def command_record_output(args: argparse.Namespace) -> int:
    print(json.dumps(asyncio.run(_record_output(args)), sort_keys=True))
    return 0


async def _ws_auth(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {_token(args.token_file)}"}

    async def authenticate() -> None:
        async with websockets.connect(
            args.url,
            ssl=_context(args.ca_file, insecure=args.insecure),
            origin=args.origin,
            max_size=args.max_bytes,
            **_websocket_headers(headers),
        ):
            return None

    try:
        await asyncio.wait_for(authenticate(), timeout=args.timeout)
        accepted = True
        status = 101
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(exc, "status_code", None)
        if status is None:
            raise
        accepted = False
    if args.expect == "accept" and not accepted:
        raise RuntimeError(f"WebSocket was rejected with status {status}")
    if args.expect == "reject" and accepted:
        raise RuntimeError("WebSocket unexpectedly accepted the credential")
    return {"accepted": accepted, "status": status}


def command_ws_auth(args: argparse.Namespace) -> int:
    print(json.dumps(asyncio.run(_ws_auth(args)), sort_keys=True))
    return 0


def command_capture_once(args: argparse.Namespace) -> int:
    output = Path(args.output)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind, args.port))
    listener.listen(1)
    listener.settimeout(args.timeout)
    captured = bytearray()
    try:
        connection, _peer = listener.accept()
        with connection:
            connection.settimeout(0.25)
            deadline = time.monotonic() + args.timeout
            while len(captured) < args.max_bytes and time.monotonic() < deadline:
                try:
                    chunk = connection.recv(
                        min(16 * 1024, args.max_bytes - len(captured))
                    )
                except TimeoutError:
                    break
                if not chunk:
                    break
                captured.extend(chunk)
    finally:
        listener.close()
    descriptor = os.open(output, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(captured)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short capture write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"captured_bytes": len(captured)}, sort_keys=True))
    return 0


def command_wait_tcp(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.wait_timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((args.host, args.port), timeout=0.5):
                print(json.dumps({"host": args.host, "port": args.port}))
                return 0
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("TCP listener did not become ready")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    def add_http(name: str, callback) -> None:
        command = commands.add_parser(name)
        command.add_argument("--url", required=True)
        command.add_argument("--token-file", required=True)
        command.add_argument("--ca-file", required=True)
        command.add_argument("--insecure", action="store_true")
        command.add_argument("--method", default="GET")
        command.add_argument("--expect-status", type=int)
        command.add_argument("--expect-code")
        command.add_argument("--timeout", type=float, default=3.0)
        if name == "wait-http":
            command.add_argument("--wait-timeout", type=float, default=30.0)
        command.set_defaults(callback=callback)

    add_http("http", command_http)
    add_http("wait-http", command_wait_http)

    browser_create = commands.add_parser("browser-session-create")
    browser_create.add_argument("--url", required=True)
    browser_create.add_argument("--token-file", required=True)
    browser_create.add_argument("--ca-file", required=True)
    browser_create.add_argument("--origin", required=True)
    browser_create.add_argument("--cookie-file", required=True)
    browser_create.add_argument("--insecure", action="store_true")
    browser_create.add_argument("--timeout", type=float, default=3.0)
    browser_create.set_defaults(callback=command_browser_session_create)

    browser_check = commands.add_parser("browser-session-check")
    browser_check.add_argument("--url", required=True)
    browser_check.add_argument("--ca-file", required=True)
    browser_check.add_argument("--origin", required=True)
    browser_check.add_argument("--cookie-file", required=True)
    browser_check.add_argument("--insecure", action="store_true")
    browser_check.add_argument("--timeout", type=float, default=3.0)
    browser_check.set_defaults(callback=command_browser_session_check)

    output = commands.add_parser("output-frame")
    output.add_argument("--url", required=True)
    output.add_argument("--token-file", required=True)
    output.add_argument("--ca-file", required=True)
    output.add_argument("--insecure", action="store_true")
    output.add_argument("--origin", required=True)
    output.add_argument("--expect", choices=("slate", "rendered"), required=True)
    output.add_argument("--timeout", type=float, default=5.0)
    output.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    output.add_argument("--slate-max-error", type=float, default=8.0)
    output.set_defaults(callback=command_output_frame)

    recorder = commands.add_parser("record-output")
    recorder.add_argument("--url", required=True)
    recorder.add_argument("--token-file", required=True)
    recorder.add_argument("--ca-file", required=True)
    recorder.add_argument("--insecure", action="store_true")
    recorder.add_argument("--origin", required=True)
    recorder.add_argument("--expect", choices=("slate", "rendered"), required=True)
    recorder.add_argument("--frames", type=int, default=6, choices=range(2, 61))
    recorder.add_argument("--timeout", type=float, default=10.0)
    recorder.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    recorder.add_argument("--slate-max-error", type=float, default=8.0)
    recorder.set_defaults(callback=command_record_output)

    ws_auth = commands.add_parser("ws-auth")
    ws_auth.add_argument("--url", required=True)
    ws_auth.add_argument("--token-file", required=True)
    ws_auth.add_argument("--ca-file", required=True)
    ws_auth.add_argument("--insecure", action="store_true")
    ws_auth.add_argument("--origin", required=True)
    ws_auth.add_argument("--expect", choices=("accept", "reject"), required=True)
    ws_auth.add_argument("--timeout", type=float, default=5.0)
    ws_auth.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    ws_auth.set_defaults(callback=command_ws_auth)

    capture = commands.add_parser("capture-once")
    capture.add_argument("--bind", default="0.0.0.0")
    capture.add_argument("--port", type=int, required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--timeout", type=float, default=15.0)
    capture.add_argument("--max-bytes", type=int, default=64 * 1024)
    capture.set_defaults(callback=command_capture_once)

    wait_tcp = commands.add_parser("wait-tcp")
    wait_tcp.add_argument("--host", required=True)
    wait_tcp.add_argument("--port", type=int, required=True)
    wait_tcp.add_argument("--wait-timeout", type=float, default=30.0)
    wait_tcp.set_defaults(callback=command_wait_tcp)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.callback(args)
    except Exception as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
