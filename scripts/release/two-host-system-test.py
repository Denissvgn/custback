#!/usr/bin/env python3
"""Packaged two-host WSS/HTTPS release-system qualification.

The host orchestrator accepts one already-built wheel, installs that exact file
into a purpose-built image, and runs the meeting and renderer processes at
distinct non-loopback addresses on a private Docker network.  Every scenario is
declared by the reviewed Phase 6 manifest and mapped to an explicit handler;
missing or extra IDs fail before Docker is mutated.

All credentials and PKI material live below a disposable run directory.  The
meeting and renderer receive different read-only bind mounts, while CA signing
keys remain on the host.  Containers, networks, image, sockets, and the entire
secret tree are removed in ``finally`` paths on success, failure, timeout, or
signal-driven cancellation.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import ipaddress
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "scripts" / "release" / "required-gates.json"
ASSET_DIRECTORY = Path(__file__).resolve().with_name("two-host")
PROBE_PATH = "/opt/custback-two-host-probe.py"
MEETING_NAME = "meeting.test"
RENDERER_NAME = "renderer.test"
MEETING_PORT = 8710
RENDERER_PORT = 8711
CAPTURE_FRAME_PORT = 9443
CAPTURE_CONTROL_PORT = 9444

TLS_SCENARIOS = (
    "nominal-rendering",
    "renderer-outage",
    "stale-renderer-output",
    "wrong-renderer-token",
    "wrong-control-token",
    "renderer-untrusted-ca",
    "renderer-expired-certificate",
    "renderer-wrong-host-certificate",
    "control-untrusted-ca",
    "control-expired-certificate",
    "control-wrong-host-certificate",
    "renderer-firewall-path-removed",
    "control-firewall-path-removed",
    "renderer-reconnect",
    "credential-and-ca-rotation",
    "privacy-slate-startup",
    "privacy-slate-frame-plane-failure",
    "control-error-mapping",
    "failed-tls-handshake-no-payload",
    "clean-host-repeat-a",
    "clean-host-repeat-b",
)

_WHEEL_RE = re.compile(
    r"^custback-[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9_.+-]*)?-py3-none-any\.whl$"
)
_SAFE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class HarnessError(RuntimeError):
    """The qualification harness or one scenario failed closed."""


class HarnessCancelled(BaseException):
    """A termination signal requested cancellation and teardown."""


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    """Bounded, shell-free command execution suitable for replacement in tests."""

    def run(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        check: bool = True,
        cwd: Path | None = None,
    ) -> CommandResult:
        argv = tuple(os.fspath(part) for part in command)
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HarnessError(
                f"command timed out after {timeout:g}s: {' '.join(argv[:4])}"
            ) from exc
        result = CommandResult(
            argv,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            tail = detail[-1] if detail else "no diagnostic output"
            raise HarnessError(
                f"command failed ({result.returncode}): {' '.join(argv[:4])}: {tail}"
            )
        return result


class DockerClient:
    def __init__(
        self,
        runner: CommandRunner,
        *,
        binary: str = "docker",
        command_timeout: float = 45.0,
    ):
        self.runner = runner
        self.binary = binary
        self.command_timeout = command_timeout

    def run(
        self,
        *arguments: str,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        return self.runner.run(
            (self.binary, *arguments),
            timeout=self.command_timeout if timeout is None else timeout,
            check=check,
        )

    def best_effort(self, *arguments: str) -> None:
        with contextlib.suppress(Exception):
            self.run(*arguments, timeout=min(self.command_timeout, 15.0), check=False)


@dataclass
class ResourceTracker:
    """Names every external resource before creation and removes it idempotently."""

    docker: DockerClient
    containers: list[str] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    cleaned: bool = False

    def expect_container(self, name: str) -> None:
        if name not in self.containers:
            self.containers.append(name)

    def expect_network(self, name: str) -> None:
        if name not in self.networks:
            self.networks.append(name)

    def expect_image(self, name: str) -> None:
        if name not in self.images:
            self.images.append(name)

    def cleanup(self) -> None:
        if self.cleaned:
            return
        for container in reversed(self.containers):
            self.docker.best_effort("rm", "-f", container)
        for network in reversed(self.networks):
            self.docker.best_effort("network", "rm", network)
        for image in reversed(self.images):
            self.docker.best_effort("image", "rm", "-f", image)
        self.cleaned = True


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    path.chmod(0o700)
    return path


def _private_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short private write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_private(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _private_write(temporary, payload)
    os.replace(temporary, path)
    path.chmod(0o600)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_private(source: Path, destination: Path) -> None:
    _private_write(destination, source.read_bytes())


def _json_private(path: Path, value: Mapping[str, Any]) -> None:
    _private_write(
        path,
        (json.dumps(value, sort_keys=False, indent=2) + "\n").encode("utf-8"),
    )


def _certificate_time(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%d%H%M%SZ")


class CertificateAuthority:
    """One ephemeral CA whose signing key is never mounted into a container."""

    def __init__(self, root: Path, name: str, runner: CommandRunner):
        if not _SAFE_NAME_RE.fullmatch(name):
            raise HarnessError(f"unsafe CA name: {name}")
        self.root = _private_directory(root / name)
        self.name = name
        self.runner = runner
        self.key = self.root / "ca.key"
        self.certificate = self.root / "ca.pem"
        self.config = self.root / "ca.cnf"
        self._initialize()

    def _openssl(self, *arguments: str) -> None:
        self.runner.run(("openssl", *arguments), timeout=30.0)

    def _initialize(self) -> None:
        _private_directory(self.root / "newcerts")
        _private_write(self.root / "index.txt", b"")
        _private_write(self.root / "serial", b"1000\n")
        self._openssl(
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-sha256",
            "-days",
            "3",
            "-subj",
            f"/CN=custback-phase6-{self.name}",
            "-keyout",
            str(self.key),
            "-out",
            str(self.certificate),
        )
        now = dt.datetime.now(dt.timezone.utc)
        config = f"""
[ca]
default_ca = phase6_ca

[phase6_ca]
database = {self.root / "index.txt"}
new_certs_dir = {self.root / "newcerts"}
serial = {self.root / "serial"}
certificate = {self.certificate}
private_key = {self.key}
default_md = sha256
default_days = 2
policy = phase6_policy
copy_extensions = copy
unique_subject = no

[phase6_policy]
commonName = supplied

[server_cert]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer

[phase6_metadata]
created = {_certificate_time(now)}
""".lstrip()
        _private_write(self.config, config.encode("utf-8"))
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(0o600)

    def issue(
        self,
        directory: Path,
        label: str,
        hostname: str,
        *,
        expired: bool = False,
    ) -> tuple[Path, Path]:
        if not _SAFE_NAME_RE.fullmatch(label) or not _SAFE_NAME_RE.fullmatch(hostname):
            raise HarnessError("unsafe certificate label or hostname")
        key = directory / f"{label}.key"
        request = directory / f"{label}.csr"
        certificate = directory / f"{label}.crt"
        self._openssl(
            "req",
            "-new",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-sha256",
            "-subj",
            f"/CN={hostname}",
            "-addext",
            f"subjectAltName=DNS:{hostname}",
            "-keyout",
            str(key),
            "-out",
            str(request),
        )
        now = dt.datetime.now(dt.timezone.utc)
        start = (
            dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
            if expired
            else now - dt.timedelta(days=1)
        )
        end = (
            dt.datetime(2000, 1, 2, tzinfo=dt.timezone.utc)
            if expired
            else now + dt.timedelta(days=2)
        )
        self._openssl(
            "ca",
            "-batch",
            "-notext",
            "-config",
            str(self.config),
            "-extensions",
            "server_cert",
            "-startdate",
            _certificate_time(start),
            "-enddate",
            _certificate_time(end),
            "-in",
            str(request),
            "-out",
            str(certificate),
        )
        request.unlink()
        key.chmod(0o600)
        certificate.chmod(0o600)
        return certificate, key


@dataclass(frozen=True)
class ScenarioOptions:
    meeting_certificate: str = "valid"  # valid | untrusted | expired | wrong-host
    renderer_certificate: str = "valid"
    renderer_token_matches: bool = True
    control_token_matches: bool = True
    source_url: str = f"wss://{MEETING_NAME}:{MEETING_PORT}"
    control_url: str = f"https://{RENDERER_NAME}:{RENDERER_PORT}"
    rotation: bool = False


@dataclass
class ScenarioMaterial:
    root: Path
    meeting: Path
    renderer: Path
    authorities: Path
    management_token: str = field(repr=False)
    renderer_token: str = field(repr=False)
    control_token: str = field(repr=False)
    renderer_token_v2: str | None = field(default=None, repr=False)
    control_token_v2: str | None = field(default=None, repr=False)
    rotation_files: dict[str, bytes] = field(default_factory=dict, repr=False)
    meeting_observer_insecure: bool = False
    renderer_observer_insecure: bool = False

    def _require_rotation(self) -> None:
        if (
            not self.rotation_files
            or self.renderer_token_v2 is None
            or self.control_token_v2 is None
        ):
            raise HarnessError("rotation material was not provisioned")

    def rotate_renderer_listener_token(self) -> None:
        self._require_rotation()
        assert self.renderer_token_v2 is not None
        payload = self.renderer_token_v2.encode() + b"\n"
        _replace_private(self.meeting / "renderer-token", payload)

    def rotate_renderer_client_token(self) -> None:
        self._require_rotation()
        assert self.renderer_token_v2 is not None
        payload = self.renderer_token_v2.encode() + b"\n"
        _replace_private(self.renderer / "renderer-token", payload)
        self.renderer_token = self.renderer_token_v2

    def rotate_control_listener_token(self) -> None:
        self._require_rotation()
        assert self.control_token_v2 is not None
        payload = self.control_token_v2.encode() + b"\n"
        _replace_private(self.renderer / "control-token", payload)

    def rotate_control_client_token(self) -> None:
        self._require_rotation()
        assert self.control_token_v2 is not None
        payload = self.control_token_v2.encode() + b"\n"
        _replace_private(self.meeting / "control-token", payload)
        self.control_token = self.control_token_v2

    @staticmethod
    def _ca_bundle(old: bytes, new: bytes) -> bytes:
        return old.rstrip() + b"\n" + new.lstrip()

    def stage_meeting_ca(self) -> None:
        self._require_rotation()
        path = self.renderer / "meeting-ca.pem"
        _replace_private(
            path,
            self._ca_bundle(path.read_bytes(), self.rotation_files["meeting-ca"]),
        )

    def activate_meeting_certificate(self) -> None:
        self._require_rotation()
        _replace_private(
            self.meeting / "server.crt", self.rotation_files["meeting-cert"]
        )
        _replace_private(
            self.meeting / "server.key", self.rotation_files["meeting-key"]
        )
        _replace_private(
            self.meeting / "server-ca.pem", self.rotation_files["meeting-ca"]
        )

    def retire_meeting_ca(self) -> None:
        self._require_rotation()
        _replace_private(
            self.renderer / "meeting-ca.pem", self.rotation_files["meeting-ca"]
        )

    def stage_renderer_ca(self) -> None:
        self._require_rotation()
        path = self.meeting / "renderer-ca.pem"
        _replace_private(
            path,
            self._ca_bundle(path.read_bytes(), self.rotation_files["renderer-ca"]),
        )

    def activate_renderer_certificate(self) -> None:
        self._require_rotation()
        _replace_private(
            self.renderer / "server.crt", self.rotation_files["renderer-cert"]
        )
        _replace_private(
            self.renderer / "server.key", self.rotation_files["renderer-key"]
        )
        _replace_private(
            self.renderer / "server-ca.pem", self.rotation_files["renderer-ca"]
        )

    def retire_renderer_ca(self) -> None:
        self._require_rotation()
        _replace_private(
            self.meeting / "renderer-ca.pem", self.rotation_files["renderer-ca"]
        )

    def install_correct_control_client(self) -> None:
        _replace_private(
            self.meeting / "control-token", self.control_token.encode() + b"\n"
        )


class MaterialFactory:
    def __init__(self, runner: CommandRunner):
        self.runner = runner

    def _certificate(
        self,
        authorities: Path,
        issued: Path,
        plane: str,
        hostname: str,
        profile: str,
    ) -> tuple[Path, Path, Path, CertificateAuthority]:
        legitimate = CertificateAuthority(authorities, f"{plane}-ca", self.runner)
        signer = legitimate
        cert_hostname = hostname
        expired = False
        if profile == "untrusted":
            signer = CertificateAuthority(authorities, f"{plane}-rogue-ca", self.runner)
        elif profile == "expired":
            expired = True
        elif profile == "wrong-host":
            cert_hostname = f"wrong-{plane}.test"
        elif profile != "valid":
            raise HarnessError(f"unknown certificate profile: {profile}")
        certificate, key = signer.issue(
            issued,
            f"{plane}-server",
            cert_hostname,
            expired=expired,
        )
        return certificate, key, signer.certificate, legitimate

    def create(
        self, root: Path, scenario_id: str, options: ScenarioOptions
    ) -> ScenarioMaterial:
        scenario_root = _private_directory(root / scenario_id)
        meeting = _private_directory(scenario_root / "meeting-secrets")
        renderer = _private_directory(scenario_root / "renderer-secrets")
        authorities = _private_directory(scenario_root / "authorities")
        issued = _private_directory(scenario_root / "issued")
        if meeting.resolve() == renderer.resolve():
            raise HarnessError("meeting and renderer secret directories must differ")

        management_token = secrets.token_urlsafe(32)
        renderer_token = secrets.token_urlsafe(32)
        control_token = secrets.token_urlsafe(32)
        wrong_renderer = secrets.token_urlsafe(32)
        wrong_control = secrets.token_urlsafe(32)

        meeting_cert, meeting_key, meeting_server_ca, meeting_ca = self._certificate(
            authorities,
            issued,
            "meeting",
            MEETING_NAME,
            options.meeting_certificate,
        )
        renderer_cert, renderer_key, renderer_server_ca, renderer_ca = (
            self._certificate(
                authorities,
                issued,
                "renderer",
                RENDERER_NAME,
                options.renderer_certificate,
            )
        )

        private_files: dict[Path, bytes] = {
            meeting / "management-token": management_token.encode() + b"\n",
            meeting / "renderer-token": renderer_token.encode() + b"\n",
            meeting / "control-token": (
                control_token if options.control_token_matches else wrong_control
            ).encode()
            + b"\n",
            meeting / "old-renderer-token": renderer_token.encode() + b"\n",
            meeting / "old-control-token": control_token.encode() + b"\n",
            renderer / "renderer-token": (
                renderer_token if options.renderer_token_matches else wrong_renderer
            ).encode()
            + b"\n",
            renderer / "control-token": control_token.encode() + b"\n",
            renderer / "old-renderer-token": renderer_token.encode() + b"\n",
            renderer / "old-control-token": control_token.encode() + b"\n",
            meeting / "server.crt": meeting_cert.read_bytes(),
            meeting / "server.key": meeting_key.read_bytes(),
            meeting / "server-ca.pem": meeting_server_ca.read_bytes(),
            meeting / "renderer-ca.pem": renderer_ca.certificate.read_bytes(),
            renderer / "server.crt": renderer_cert.read_bytes(),
            renderer / "server.key": renderer_key.read_bytes(),
            renderer / "server-ca.pem": renderer_server_ca.read_bytes(),
            renderer / "meeting-ca.pem": meeting_ca.certificate.read_bytes(),
        }
        for path, payload in private_files.items():
            _private_write(path, payload)

        material = ScenarioMaterial(
            scenario_root,
            meeting,
            renderer,
            authorities,
            management_token,
            renderer_token,
            control_token,
            meeting_observer_insecure=options.meeting_certificate
            in {"expired", "wrong-host"},
            renderer_observer_insecure=options.renderer_certificate
            in {"expired", "wrong-host"},
        )
        if options.rotation:
            material.renderer_token_v2 = secrets.token_urlsafe(32)
            material.control_token_v2 = secrets.token_urlsafe(32)
            meeting_ca_v2 = CertificateAuthority(
                authorities, "meeting-ca-v2", self.runner
            )
            renderer_ca_v2 = CertificateAuthority(
                authorities, "renderer-ca-v2", self.runner
            )
            meeting_cert_v2, meeting_key_v2 = meeting_ca_v2.issue(
                issued, "meeting-server-v2", MEETING_NAME
            )
            renderer_cert_v2, renderer_key_v2 = renderer_ca_v2.issue(
                issued, "renderer-server-v2", RENDERER_NAME
            )
            material.rotation_files = {
                "meeting-cert": meeting_cert_v2.read_bytes(),
                "meeting-key": meeting_key_v2.read_bytes(),
                "meeting-ca": meeting_ca_v2.certificate.read_bytes(),
                "renderer-cert": renderer_cert_v2.read_bytes(),
                "renderer-key": renderer_key_v2.read_bytes(),
                "renderer-ca": renderer_ca_v2.certificate.read_bytes(),
            }

        meeting_config = {
            "camera": {
                "device": 0,
                "width": 320,
                "height": 180,
                "fps": 10,
                "synthetic": True,
            },
            "background": {"mode": "remote", "remote_fallback_mode": "color"},
            "segmentation": {"backend": "none"},
            "output": {"backend": "null", "fps": 10, "preview": False},
            "api": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": MEETING_PORT,
                "remote_timeout_ms": 250,
                "allow_non_loopback": True,
                "allowed_origins": [f"https://{MEETING_NAME}:{MEETING_PORT}"],
                "token_file": "/run/custback/management-token",
                "renderer_token_file": "/run/custback/renderer-token",
                "tls_certfile": "/run/custback/server.crt",
                "tls_keyfile": "/run/custback/server.key",
            },
            "avatar": {
                "url": options.control_url,
                "token_file": "/run/custback/control-token",
                "tls_ca_file": "/run/custback/renderer-ca.pem",
                "connect_timeout_s": 1.0,
                "read_timeout_s": 3.0,
            },
        }
        avatar_config = {
            "source": {
                "url": options.source_url,
                "token_file": "/run/custback/renderer-token",
                "tls_ca_file": "/run/custback/meeting-ca.pem",
                "connect_timeout_s": 1.0,
                "reconnect_min_s": 0.1,
                "reconnect_max_s": 0.5,
            },
            "driver": {"backend": "idle"},
            "render": {"max_fps": 10, "jpeg_quality": 85},
            "storage": {
                "rigs_dir": "/var/lib/custback/rigs",
                "backgrounds_dir": "/var/lib/custback/avatar-backgrounds",
            },
            "api": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": RENDERER_PORT,
                "token_file": "/run/custback/control-token",
                "allow_non_loopback": True,
                "allowed_origins": [
                    f"https://{RENDERER_NAME}:{RENDERER_PORT}",
                    f"https://{MEETING_NAME}:{MEETING_PORT}",
                ],
                "tls_certfile": "/run/custback/server.crt",
                "tls_keyfile": "/run/custback/server.key",
            },
        }
        _json_private(meeting / "config.yaml", meeting_config)
        _json_private(renderer / "avatar.yaml", avatar_config)
        return material


@dataclass
class ScenarioResult:
    scenario_id: str
    duration_s: float
    meeting_address: str
    renderer_address: str


class ScenarioStack:
    def __init__(
        self,
        context: "HarnessContext",
        scenario_id: str,
        options: ScenarioOptions = ScenarioOptions(),
    ):
        self.context = context
        self.scenario_id = scenario_id
        self.options = options
        suffix = hashlib.sha256(f"{context.run_id}:{scenario_id}".encode()).hexdigest()[
            :10
        ]
        self.network = f"custback-p6-{suffix}"
        self.meeting_container = f"custback-p6-meeting-{suffix}"
        self.renderer_container = f"custback-p6-renderer-{suffix}"
        self.tracker = ResourceTracker(context.docker)
        self.material = context.materials.create(context.run_root, scenario_id, options)
        self.meeting_address = ""
        self.renderer_address = ""
        self.started_meeting = False
        self.started_renderer = False
        self._blocked: set[tuple[str, int]] = set()
        self._recording_count = 0
        self._browser_session_created = False

    @property
    def label(self) -> str:
        return f"custback.phase6.run={self.context.run_id}"

    def __enter__(self) -> "ScenarioStack":
        self.tracker.expect_network(self.network)
        self.context.docker.run(
            "network",
            "create",
            "--internal",
            "--label",
            self.label,
            self.network,
        )
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        try:
            self._collect_logs()
        finally:
            self.tracker.cleanup()

    def _collect_logs(self) -> None:
        for name, label in (
            (self.meeting_container, "meeting"),
            (self.renderer_container, "renderer"),
        ):
            if name not in self.tracker.containers:
                continue
            try:
                result = self.context.docker.run(
                    "logs", name, check=False, timeout=10.0
                )
            except Exception:
                continue
            log_path = self.material.root / f"{label}.log"
            with contextlib.suppress(Exception):
                _replace_private(
                    log_path,
                    (result.stdout + result.stderr).encode("utf-8", errors="replace"),
                )

    def _container_arguments(
        self,
        *,
        name: str,
        hostname: str,
        secrets_directory: Path,
    ) -> list[str]:
        return [
            "run",
            "-d",
            "--name",
            name,
            "--hostname",
            hostname,
            "--network",
            self.network,
            "--network-alias",
            hostname,
            "--label",
            self.label,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "NET_ADMIN",
            "--security-opt",
            "no-new-privileges",
            "--env",
            "HOME=/var/lib/custback/home",
            "--mount",
            f"type=bind,src={secrets_directory},dst=/run/custback,readonly",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,mode=0700",
            "--tmpfs",
            "/var/lib/custback:rw,nosuid,nodev,mode=0700",
            self.context.image,
        ]

    def start_meeting(self) -> None:
        if self.started_meeting:
            return
        self.tracker.expect_container(self.meeting_container)
        command = self._container_arguments(
            name=self.meeting_container,
            hostname=MEETING_NAME,
            secrets_directory=self.material.meeting,
        )
        command.extend(["custback", "-c", "/run/custback/config.yaml", "--no-file-log"])
        self.context.docker.run(*command, timeout=self.context.command_timeout)
        self.started_meeting = True
        self._wait_tcp(self.meeting_container, MEETING_NAME, MEETING_PORT)
        self._wait_core_ready()
        self._refresh_addresses()

    def start_renderer(self) -> None:
        if self.started_renderer:
            return
        self.tracker.expect_container(self.renderer_container)
        command = self._container_arguments(
            name=self.renderer_container,
            hostname=RENDERER_NAME,
            secrets_directory=self.material.renderer,
        )
        command.extend(["custback", "avatar", "-c", "/run/custback/avatar.yaml"])
        self.context.docker.run(*command, timeout=self.context.command_timeout)
        self.started_renderer = True
        self._wait_tcp(self.renderer_container, RENDERER_NAME, RENDERER_PORT)
        self._wait_avatar_ready()
        self._refresh_addresses()

    def start(self, *, renderer: bool = True) -> None:
        self.start_meeting()
        if renderer:
            self.start_renderer()

    def _refresh_addresses(self) -> None:
        if self.started_meeting:
            self.meeting_address = self._address(self.meeting_container)
        if self.started_renderer:
            self.renderer_address = self._address(self.renderer_container)
        if self.meeting_address and self.renderer_address:
            meeting = ipaddress.ip_address(self.meeting_address)
            renderer = ipaddress.ip_address(self.renderer_address)
            if meeting.is_loopback or renderer.is_loopback or meeting == renderer:
                raise HarnessError(
                    "two-host containers do not have distinct non-loopback addresses"
                )

    def _address(self, container: str) -> str:
        result = self.context.docker.run(
            "inspect",
            "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            container,
        )
        address = result.stdout.strip()
        try:
            ipaddress.ip_address(address)
        except ValueError as exc:
            raise HarnessError(
                f"container has no isolated address: {container}"
            ) from exc
        return address

    def _probe(
        self, container: str, *arguments: str, timeout: float | None = None
    ) -> dict[str, Any]:
        result = self.context.docker.run(
            "exec",
            container,
            "python",
            PROBE_PATH,
            *arguments,
            timeout=self.context.command_timeout if timeout is None else timeout,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return {}
        try:
            parsed = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise HarnessError("container probe did not return JSON") from exc
        if not isinstance(parsed, dict):
            raise HarnessError("container probe returned a non-object")
        return parsed

    def _wait_tcp(self, container: str, host: str, port: int) -> None:
        self._probe(
            container,
            "wait-tcp",
            "--host",
            host,
            "--port",
            str(port),
            "--wait-timeout",
            str(self.context.scenario_timeout),
            timeout=self.context.scenario_timeout + 5.0,
        )

    def _http(
        self,
        container: str,
        url: str,
        token_file: str,
        ca_file: str,
        *,
        expect_status: int | None = None,
        expect_code: str | None = None,
        insecure: bool = False,
        wait: bool = False,
    ) -> dict[str, Any]:
        arguments = [
            "wait-http" if wait else "http",
            "--url",
            url,
            "--token-file",
            token_file,
            "--ca-file",
            ca_file,
            "--timeout",
            "2",
        ]
        if insecure:
            arguments.append("--insecure")
        if wait:
            arguments.extend(["--wait-timeout", str(self.context.scenario_timeout)])
        if expect_status is not None:
            arguments.extend(["--expect-status", str(expect_status)])
        if expect_code is not None:
            arguments.extend(["--expect-code", expect_code])
        return self._probe(
            container,
            *arguments,
            timeout=self.context.scenario_timeout + 5.0 if wait else None,
        )

    def _wait_core_ready(self) -> None:
        self._http(
            self.meeting_container,
            f"https://{MEETING_NAME}:{MEETING_PORT}/status",
            "/run/custback/management-token",
            "/run/custback/server-ca.pem",
            expect_status=200,
            insecure=self.material.meeting_observer_insecure,
            wait=True,
        )

    def _wait_avatar_ready(self) -> None:
        self._http(
            self.renderer_container,
            f"https://{RENDERER_NAME}:{RENDERER_PORT}/status",
            "/run/custback/control-token",
            "/run/custback/server-ca.pem",
            expect_status=200,
            insecure=self.material.renderer_observer_insecure,
            wait=True,
        )

    def core_status(self) -> dict[str, Any]:
        result = self._http(
            self.meeting_container,
            f"https://{MEETING_NAME}:{MEETING_PORT}/status",
            "/run/custback/management-token",
            "/run/custback/server-ca.pem",
            expect_status=200,
            insecure=self.material.meeting_observer_insecure,
        )
        body = result.get("body")
        return body if isinstance(body, dict) else {}

    def avatar_status(self) -> dict[str, Any]:
        result = self._http(
            self.renderer_container,
            f"https://{RENDERER_NAME}:{RENDERER_PORT}/status",
            "/run/custback/control-token",
            "/run/custback/server-ca.pem",
            expect_status=200,
            insecure=self.material.renderer_observer_insecure,
        )
        body = result.get("body")
        return body if isinstance(body, dict) else {}

    def proxy_status(self, *, code: str | None = None) -> dict[str, Any]:
        return self._http(
            self.meeting_container,
            f"https://{MEETING_NAME}:{MEETING_PORT}/avatar/status",
            "/run/custback/management-token",
            "/run/custback/server-ca.pem",
            expect_status=502 if code else 200,
            expect_code=code,
            insecure=self.material.meeting_observer_insecure,
        )

    def direct_avatar(
        self,
        *,
        token_file: str = "/run/custback/control-token",
        expect_status: int = 200,
        expect_code: str | None = None,
    ) -> dict[str, Any]:
        return self._http(
            self.meeting_container,
            f"https://{RENDERER_NAME}:{RENDERER_PORT}/status",
            token_file,
            "/run/custback/renderer-ca.pem",
            expect_status=expect_status,
            expect_code=expect_code,
        )

    def establish_browser_session(self) -> None:
        arguments = [
            "browser-session-create",
            "--url",
            f"https://{MEETING_NAME}:{MEETING_PORT}/auth/session",
            "--token-file",
            "/run/custback/management-token",
            "--ca-file",
            "/run/custback/server-ca.pem",
            "--origin",
            f"https://{MEETING_NAME}:{MEETING_PORT}",
            "--cookie-file",
            "/tmp/custback-browser-session",
        ]
        if self.material.meeting_observer_insecure:
            arguments.append("--insecure")
        self._probe(self.meeting_container, *arguments)
        self._browser_session_created = True

    def assert_browser_session(self) -> None:
        if not self._browser_session_created:
            raise HarnessError("browser session was not established before the failure")
        arguments = [
            "browser-session-check",
            "--url",
            f"https://{MEETING_NAME}:{MEETING_PORT}/status",
            "--ca-file",
            "/run/custback/server-ca.pem",
            "--origin",
            f"https://{MEETING_NAME}:{MEETING_PORT}",
            "--cookie-file",
            "/tmp/custback-browser-session",
        ]
        if self.material.meeting_observer_insecure:
            arguments.append("--insecure")
        self._probe(self.meeting_container, *arguments)

    def output_frame(self, expectation: str) -> dict[str, Any]:
        arguments = [
            "output-frame",
            "--url",
            f"wss://{MEETING_NAME}:{MEETING_PORT}/ws/frames?stream=output",
            "--token-file",
            "/run/custback/management-token",
            "--ca-file",
            "/run/custback/server-ca.pem",
            "--origin",
            f"https://{MEETING_NAME}:{MEETING_PORT}",
            "--expect",
            expectation,
        ]
        if self.material.meeting_observer_insecure:
            arguments.append("--insecure")
        return self._probe(self.meeting_container, *arguments)

    def record_output(self, expectation: str, *, frames: int = 6) -> dict[str, Any]:
        """Record a bounded run of consecutive preview-publication frames."""

        arguments = [
            "record-output",
            "--url",
            f"wss://{MEETING_NAME}:{MEETING_PORT}/ws/frames?stream=output",
            "--token-file",
            "/run/custback/management-token",
            "--ca-file",
            "/run/custback/server-ca.pem",
            "--origin",
            f"https://{MEETING_NAME}:{MEETING_PORT}",
            "--expect",
            expectation,
            "--frames",
            str(frames),
        ]
        if self.material.meeting_observer_insecure:
            arguments.append("--insecure")
        result = self._probe(
            self.meeting_container,
            *arguments,
            timeout=self.context.scenario_timeout,
        )
        if result.get("frames") != frames:
            raise HarnessError("preview recorder returned an incomplete frame run")
        recording_root = self.material.root / "recordings"
        if not recording_root.exists():
            _private_directory(recording_root)
        self._recording_count += 1
        _json_private(
            recording_root / f"{self._recording_count:02d}-{expectation}-preview.json",
            result,
        )
        return result

    def ws_auth(self, token_file: str, expectation: str) -> dict[str, Any]:
        return self._probe(
            self.renderer_container,
            "ws-auth",
            "--url",
            f"wss://{MEETING_NAME}:{MEETING_PORT}/ws/frames?stream=raw",
            "--token-file",
            token_file,
            "--ca-file",
            "/run/custback/meeting-ca.pem",
            "--origin",
            f"https://{MEETING_NAME}:{MEETING_PORT}",
            "--expect",
            expectation,
        )

    def poll(
        self,
        callback: Callable[[], dict[str, Any]],
        predicate: Callable[[dict[str, Any]], bool],
        description: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.context.scenario_timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = callback()
                if predicate(last):
                    return last
            except HarnessError:
                pass
            time.sleep(0.2)
        raise HarnessError(f"timed out waiting for {description}: {last}")

    def wait_nominal(self, *, previous_frames: int = -1) -> dict[str, Any]:
        core = self.poll(
            self.core_status,
            lambda value: (
                bool(value.get("remote_connected"))
                and int(value.get("remote_frames_used", 0)) > previous_frames
            ),
            "authenticated renderer frames",
        )
        self.poll(
            self.avatar_status,
            lambda value: (
                bool(value.get("connected"))
                and int(value.get("frames_received", 0)) > 0
                and int(value.get("frames_rendered", 0)) > 0
                and int(value.get("frames_sent", 0)) > 0
                and value.get("driver_backend") == "idle"
            ),
            "packaged idle renderer output",
        )
        self.record_output("rendered")
        return core

    def wait_privacy_slate(self, *reasons: str) -> dict[str, Any]:
        accepted = set(reasons)
        core = self.poll(
            self.core_status,
            lambda value: (
                bool(value.get("remote_fallback_active"))
                and (
                    not accepted
                    or str(value.get("remote_fallback_reason", "")) in accepted
                )
            ),
            "privacy-slate fallback",
        )
        self.record_output("slate")
        return core

    def block(self, container: str, port: int) -> None:
        name = (
            self.meeting_container
            if container == "meeting"
            else self.renderer_container
        )
        self.context.docker.run(
            "exec",
            name,
            "iptables",
            "-I",
            "OUTPUT",
            "1",
            "-p",
            "tcp",
            "--dport",
            str(port),
            "-j",
            "REJECT",
        )
        self._blocked.add((container, port))

    def unblock(self, container: str, port: int) -> None:
        name = (
            self.meeting_container
            if container == "meeting"
            else self.renderer_container
        )
        self.context.docker.run(
            "exec",
            name,
            "iptables",
            "-D",
            "OUTPUT",
            "-p",
            "tcp",
            "--dport",
            str(port),
            "-j",
            "REJECT",
        )
        self._blocked.discard((container, port))

    def restart_meeting(self) -> None:
        self.context.docker.run("restart", self.meeting_container)
        self._browser_session_created = False
        self._wait_tcp(self.meeting_container, MEETING_NAME, MEETING_PORT)
        self._wait_core_ready()

    def restart_renderer(self) -> None:
        self.context.docker.run("restart", self.renderer_container)
        self._wait_tcp(self.renderer_container, RENDERER_NAME, RENDERER_PORT)
        self._wait_avatar_ready()

    def restart_both(self) -> None:
        self.context.docker.run("restart", self.meeting_container)
        self.context.docker.run("restart", self.renderer_container)
        self._wait_tcp(self.meeting_container, MEETING_NAME, MEETING_PORT)
        self._wait_core_ready()
        self._wait_tcp(self.renderer_container, RENDERER_NAME, RENDERER_PORT)
        self._wait_avatar_ready()

    def stop_renderer(self) -> None:
        self.context.docker.run("stop", "--time", "3", self.renderer_container)

    def pause_renderer(self) -> None:
        self.context.docker.run("pause", self.renderer_container)

    def unpause_renderer(self) -> None:
        self.context.docker.run("unpause", self.renderer_container, check=False)

    def start_capture(self, alias: str, port: int, output: Path) -> str:
        suffix = hashlib.sha256(
            f"{self.context.run_id}:{self.scenario_id}:{alias}".encode()
        ).hexdigest()[:8]
        name = f"custback-p6-capture-{suffix}"
        self.tracker.expect_container(name)
        self.context.docker.run(
            "run",
            "-d",
            "--name",
            name,
            "--network",
            self.network,
            "--network-alias",
            alias,
            "--label",
            self.label,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={output.parent},dst=/capture",
            self.context.image,
            "python",
            PROBE_PATH,
            "capture-once",
            "--port",
            str(port),
            "--output",
            f"/capture/{output.name}",
            "--timeout",
            str(self.context.scenario_timeout),
        )
        return name


ScenarioHandler = Callable[[ScenarioStack], None]
SCENARIO_HANDLERS: dict[str, ScenarioHandler] = {}


def scenario(identifier: str) -> Callable[[ScenarioHandler], ScenarioHandler]:
    def register(callback: ScenarioHandler) -> ScenarioHandler:
        if identifier in SCENARIO_HANDLERS:
            raise RuntimeError(f"duplicate TLS scenario handler: {identifier}")
        SCENARIO_HANDLERS[identifier] = callback
        return callback

    return register


def _nominal(stack: ScenarioStack) -> dict[str, Any]:
    stack.start()
    core = stack.wait_nominal()
    stack.proxy_status()
    return core


@scenario("nominal-rendering")
def _scenario_nominal(stack: ScenarioStack) -> None:
    _nominal(stack)


@scenario("renderer-outage")
def _scenario_renderer_outage(stack: ScenarioStack) -> None:
    _nominal(stack)
    stack.stop_renderer()
    stack.wait_privacy_slate("no-client", "stale")


@scenario("stale-renderer-output")
def _scenario_stale_output(stack: ScenarioStack) -> None:
    _nominal(stack)
    stack.pause_renderer()
    try:
        stack.wait_privacy_slate("stale")
    finally:
        stack.unpause_renderer()


@scenario("wrong-renderer-token")
def _scenario_wrong_renderer_token(stack: ScenarioStack) -> None:
    stack.start()
    stack.wait_privacy_slate("no-client", "stale")
    stack.poll(
        stack.avatar_status,
        lambda value: (
            int(value.get("connect_attempts", 0)) >= 1
            and not bool(value.get("connected"))
        ),
        "wrong renderer-token rejection",
    )
    stack.ws_auth("/run/custback/renderer-token", "reject")


@scenario("wrong-control-token")
def _scenario_wrong_control(stack: ScenarioStack) -> None:
    stack.start()
    stack.wait_nominal()
    stack.establish_browser_session()
    stack.proxy_status(code="avatar_auth_failed")
    stack.assert_browser_session()
    if not stack.core_status().get("remote_connected"):
        raise HarnessError("control-token failure invalidated the frame session")


def _renderer_certificate_failure(stack: ScenarioStack) -> None:
    stack.start()
    stack.wait_privacy_slate("no-client", "stale")
    stack.poll(
        stack.avatar_status,
        lambda value: (
            int(value.get("connect_attempts", 0)) >= 1
            and not bool(value.get("connected"))
        ),
        "renderer TLS verification failure",
    )


@scenario("renderer-untrusted-ca")
def _scenario_renderer_untrusted(stack: ScenarioStack) -> None:
    _renderer_certificate_failure(stack)


@scenario("renderer-expired-certificate")
def _scenario_renderer_expired(stack: ScenarioStack) -> None:
    _renderer_certificate_failure(stack)


@scenario("renderer-wrong-host-certificate")
def _scenario_renderer_wrong_host(stack: ScenarioStack) -> None:
    _renderer_certificate_failure(stack)


def _control_certificate_failure(stack: ScenarioStack) -> None:
    stack.start()
    stack.wait_nominal()
    stack.establish_browser_session()
    stack.proxy_status(code="avatar_unreachable")
    stack.assert_browser_session()
    if not stack.core_status().get("remote_connected"):
        raise HarnessError("control TLS failure invalidated the frame session")


@scenario("control-untrusted-ca")
def _scenario_control_untrusted(stack: ScenarioStack) -> None:
    _control_certificate_failure(stack)


@scenario("control-expired-certificate")
def _scenario_control_expired(stack: ScenarioStack) -> None:
    _control_certificate_failure(stack)


@scenario("control-wrong-host-certificate")
def _scenario_control_wrong_host(stack: ScenarioStack) -> None:
    _control_certificate_failure(stack)


@scenario("renderer-firewall-path-removed")
def _scenario_renderer_firewall(stack: ScenarioStack) -> None:
    _nominal(stack)
    stack.block("renderer", MEETING_PORT)
    stack.wait_privacy_slate("stale", "no-client")


@scenario("control-firewall-path-removed")
def _scenario_control_firewall(stack: ScenarioStack) -> None:
    _nominal(stack)
    stack.establish_browser_session()
    stack.block("meeting", RENDERER_PORT)
    stack.proxy_status(code="avatar_unreachable")
    stack.assert_browser_session()
    if not stack.core_status().get("remote_connected"):
        raise HarnessError("control firewall removal invalidated the frame plane")


@scenario("renderer-reconnect")
def _scenario_renderer_reconnect(stack: ScenarioStack) -> None:
    before = _nominal(stack)
    frames = int(before.get("remote_frames_used", 0))
    stack.block("renderer", MEETING_PORT)
    stack.wait_privacy_slate("stale", "no-client")
    stack.unblock("renderer", MEETING_PORT)
    stack.wait_nominal(previous_frames=frames)
    stack.poll(
        stack.avatar_status,
        lambda value: int(value.get("reconnects", 0)) >= 1,
        "renderer reconnect counter",
    )


@scenario("credential-and-ca-rotation")
def _scenario_rotation(stack: ScenarioStack) -> None:
    _nominal(stack)

    stack.material.rotate_renderer_listener_token()
    stack.restart_meeting()
    stack.wait_privacy_slate("no-client", "stale")
    stack.material.rotate_renderer_client_token()
    stack.restart_renderer()
    stack.wait_nominal()
    stack.ws_auth("/run/custback/old-renderer-token", "reject")

    stack.material.rotate_control_listener_token()
    stack.restart_renderer()
    stack.wait_nominal()
    stack.proxy_status(code="avatar_auth_failed")
    stack.material.rotate_control_client_token()
    stack.restart_meeting()
    stack.wait_nominal()
    stack.proxy_status()
    stack.direct_avatar(
        token_file="/run/custback/old-control-token",
        expect_status=401,
        expect_code="unauthorized",
    )

    stack.material.stage_meeting_ca()
    stack.restart_renderer()
    stack.wait_nominal()
    stack.material.activate_meeting_certificate()
    stack.restart_meeting()
    stack.wait_nominal()
    stack.material.retire_meeting_ca()
    stack.restart_renderer()
    stack.wait_nominal()

    stack.material.stage_renderer_ca()
    stack.restart_meeting()
    stack.wait_nominal()
    stack.proxy_status()
    stack.material.activate_renderer_certificate()
    stack.restart_renderer()
    stack.wait_nominal()
    stack.proxy_status()
    stack.material.retire_renderer_ca()
    stack.restart_meeting()
    stack.wait_nominal()
    stack.proxy_status()


@scenario("privacy-slate-startup")
def _scenario_startup_slate(stack: ScenarioStack) -> None:
    stack.start(renderer=False)
    stack.wait_privacy_slate("no-client", "stale", "startup-slate")
    stack.start_renderer()
    stack.wait_nominal()


@scenario("privacy-slate-frame-plane-failure")
def _scenario_frame_failure_slate(stack: ScenarioStack) -> None:
    _nominal(stack)
    stack.block("renderer", MEETING_PORT)
    stack.wait_privacy_slate("stale", "no-client")


@scenario("control-error-mapping")
def _scenario_control_mapping(stack: ScenarioStack) -> None:
    stack.start()
    stack.wait_nominal()
    stack.establish_browser_session()
    stack.proxy_status(code="avatar_auth_failed")
    stack.assert_browser_session()
    if stack.core_status().get("remote_connected") is not True:
        raise HarnessError("browser management authentication was invalidated")
    stack.material.install_correct_control_client()
    stack.restart_meeting()
    stack.wait_nominal()
    stack.establish_browser_session()
    stack.block("meeting", RENDERER_PORT)
    stack.proxy_status(code="avatar_unreachable")
    stack.assert_browser_session()
    if stack.core_status().get("remote_connected") is not True:
        raise HarnessError("control outage invalidated the browser/frame session")


def assert_tls_handshake_only(captured: bytes, forbidden: Iterable[bytes]) -> None:
    """Prove the untrusted peer received TLS records but no application data."""

    if not captured:
        raise HarnessError("failed TLS peer captured no ClientHello")
    for secret in forbidden:
        if secret and secret in captured:
            raise HarnessError(
                "failed TLS handshake exposed a credential or payload marker"
            )
    offset = 0
    records = 0
    while offset < len(captured):
        if len(captured) - offset < 5:
            raise HarnessError("captured TLS record header is truncated")
        content_type = captured[offset]
        length = int.from_bytes(captured[offset + 3 : offset + 5], "big")
        end = offset + 5 + length
        if end > len(captured):
            raise HarnessError("captured TLS record is truncated")
        if content_type == 23:
            raise HarnessError("TLS application data preceded peer authentication")
        if content_type not in {20, 21, 22}:
            raise HarnessError(
                f"unexpected pre-authentication TLS record {content_type}"
            )
        records += 1
        offset = end
    if records == 0:
        raise HarnessError("failed TLS peer captured no complete record")


@scenario("failed-tls-handshake-no-payload")
def _scenario_no_pre_tls_payload(stack: ScenarioStack) -> None:
    capture_root = _private_directory(stack.material.root / "captures")
    frame_capture = capture_root / "frame-plane.bin"
    control_capture = capture_root / "control-plane.bin"
    frame_container = stack.start_capture(
        "meeting-capture.test", CAPTURE_FRAME_PORT, frame_capture
    )
    control_container = stack.start_capture(
        "renderer-capture.test", CAPTURE_CONTROL_PORT, control_capture
    )
    stack.start()
    stack.proxy_status(code="avatar_unreachable")
    stack.context.docker.run(
        "wait", frame_container, timeout=stack.context.scenario_timeout
    )
    stack.context.docker.run(
        "wait", control_container, timeout=stack.context.scenario_timeout
    )
    forbidden = (
        stack.material.renderer_token.encode(),
        stack.material.control_token.encode(),
        b"Authorization: Bearer",
        b"CUSTBACK_PHASE6_PCM",
    )
    assert_tls_handshake_only(frame_capture.read_bytes(), forbidden)
    assert_tls_handshake_only(control_capture.read_bytes(), forbidden)
    stack.wait_privacy_slate("no-client", "stale")


@scenario("clean-host-repeat-a")
def _scenario_clean_repeat_a(stack: ScenarioStack) -> None:
    _nominal(stack)


@scenario("clean-host-repeat-b")
def _scenario_clean_repeat_b(stack: ScenarioStack) -> None:
    _nominal(stack)


SCENARIO_OPTIONS: dict[str, ScenarioOptions] = {
    "wrong-renderer-token": ScenarioOptions(renderer_token_matches=False),
    "wrong-control-token": ScenarioOptions(control_token_matches=False),
    "renderer-untrusted-ca": ScenarioOptions(meeting_certificate="untrusted"),
    "renderer-expired-certificate": ScenarioOptions(meeting_certificate="expired"),
    "renderer-wrong-host-certificate": ScenarioOptions(
        meeting_certificate="wrong-host"
    ),
    "control-untrusted-ca": ScenarioOptions(renderer_certificate="untrusted"),
    "control-expired-certificate": ScenarioOptions(renderer_certificate="expired"),
    "control-wrong-host-certificate": ScenarioOptions(
        renderer_certificate="wrong-host"
    ),
    "credential-and-ca-rotation": ScenarioOptions(rotation=True),
    "control-error-mapping": ScenarioOptions(control_token_matches=False),
    "failed-tls-handshake-no-payload": ScenarioOptions(
        source_url=f"wss://meeting-capture.test:{CAPTURE_FRAME_PORT}",
        control_url=f"https://renderer-capture.test:{CAPTURE_CONTROL_PORT}",
    ),
}


def load_tls_scenarios(manifest_path: Path = DEFAULT_MANIFEST) -> tuple[str, ...]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(
            f"cannot read Phase 6 gate manifest: {manifest_path}"
        ) from exc
    scenarios = manifest.get("tls_scenarios") if isinstance(manifest, dict) else None
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or any(not isinstance(item, str) or not item for item in scenarios)
        or len(scenarios) != len(set(scenarios))
    ):
        raise HarnessError("tls_scenarios must be a non-empty unique string list")
    observed = tuple(scenarios)
    manifest_set = set(observed)
    implemented_set = set(SCENARIO_HANDLERS)
    missing = sorted(manifest_set - implemented_set)
    extra = sorted(implemented_set - manifest_set)
    if missing or extra:
        raise HarnessError(
            "TLS scenario implementation mismatch; "
            f"missing={missing or 'none'} extra={extra or 'none'}"
        )
    if observed != TLS_SCENARIOS:
        raise HarnessError("reviewed TLS scenario order changed unexpectedly")
    return observed


def validate_wheel(path: Path) -> tuple[Path, str]:
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise HarnessError(f"wheel is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise HarnessError("wheel must be a regular non-symlink file")
    if not _WHEEL_RE.fullmatch(resolved.name) or not zipfile.is_zipfile(resolved):
        raise HarnessError("wheel filename or archive format is invalid")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return resolved, digest


def wheel_from_artifacts(directory: Path) -> Path:
    """Select the one reviewed wheel from a downloaded candidate directory."""

    try:
        metadata = directory.lstat()
        resolved = directory.resolve(strict=True)
    except OSError as exc:
        raise HarnessError(f"artifact directory is unavailable: {directory}") from exc
    if directory.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise HarnessError("artifact path must be a real non-symlink directory")
    candidates = [
        entry for entry in resolved.iterdir() if _WHEEL_RE.fullmatch(entry.name)
    ]
    if len(candidates) != 1:
        names = sorted(entry.name for entry in candidates)
        raise HarnessError(
            "artifact directory must contain exactly one canonical custback wheel; "
            f"found={names or 'none'}"
        )
    candidate = candidates[0]
    metadata = candidate.lstat()
    if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise HarnessError("candidate wheel must be a real non-symlink file")
    return candidate


@dataclass
class HarnessContext:
    docker: DockerClient
    runner: CommandRunner
    run_root: Path
    run_id: str
    image: str
    material_factory: MaterialFactory
    command_timeout: float
    scenario_timeout: float

    @property
    def materials(self) -> MaterialFactory:
        return self.material_factory


class TwoHostHarness:
    def __init__(
        self,
        wheel: Path,
        *,
        manifest: Path = DEFAULT_MANIFEST,
        docker_binary: str = "docker",
        command_timeout: float = 120.0,
        scenario_timeout: float = 30.0,
        temporary_base: Path | None = None,
        runner: CommandRunner | None = None,
        scenario_ids: Sequence[str] | None = None,
        clean_host_evidence: bool = False,
    ):
        self.wheel, self.wheel_sha256 = validate_wheel(wheel)
        reviewed_scenarios = load_tls_scenarios(manifest)
        if scenario_ids is None:
            self.scenarios = reviewed_scenarios
        else:
            selected = tuple(scenario_ids)
            if not selected or len(selected) != len(set(selected)):
                raise HarnessError(
                    "selected TLS scenarios must be non-empty and unique"
                )
            unknown = sorted(set(selected) - set(reviewed_scenarios))
            if unknown:
                raise HarnessError(f"unknown TLS scenario selection: {unknown}")
            expected_order = tuple(
                item for item in reviewed_scenarios if item in selected
            )
            if selected != expected_order:
                raise HarnessError("selected TLS scenarios are not in reviewed order")
            self.scenarios = selected
        clean_scenarios = {"clean-host-repeat-a", "clean-host-repeat-b"}
        if clean_host_evidence and (
            len(self.scenarios) != 1 or self.scenarios[0] not in clean_scenarios
        ):
            raise HarnessError(
                "--clean-host-evidence requires one reviewed clean-host repeat scenario"
            )
        self.clean_host_evidence = clean_host_evidence
        self.runner = runner or CommandRunner()
        self.docker = DockerClient(
            self.runner,
            binary=docker_binary,
            command_timeout=command_timeout,
        )
        self.command_timeout = command_timeout
        self.scenario_timeout = scenario_timeout
        self.temporary_base = temporary_base
        self.run_id = uuid.uuid4().hex[:16]
        self.image = f"custback-phase6-two-host:{self.run_id}"
        self.tracker = ResourceTracker(self.docker)

    def _build_image(self, run_root: Path) -> None:
        context = _private_directory(run_root / "build-context")
        shutil.copyfile(ASSET_DIRECTORY / "Dockerfile", context / "Dockerfile")
        shutil.copyfile(ASSET_DIRECTORY / "probe.py", context / "probe.py")
        shutil.copyfile(self.wheel, context / self.wheel.name)
        self.tracker.expect_image(self.image)
        self.docker.run(
            "build",
            "--label",
            f"custback.phase6.run={self.run_id}",
            "--build-arg",
            f"CUSTBACK_WHEEL={self.wheel.name}",
            "--tag",
            self.image,
            str(context),
            timeout=max(self.command_timeout, 15 * 60.0),
        )

    def _sweep_run_resources(self) -> None:
        containers: tuple[str, ...] = ()
        networks: tuple[str, ...] = ()
        try:
            result = self.docker.run(
                "ps",
                "-aq",
                "--filter",
                f"label=custback.phase6.run={self.run_id}",
                check=False,
            )
            containers = tuple(result.stdout.split())
        except Exception:
            pass
        for container in containers:
            self.docker.best_effort("rm", "-f", container)
        try:
            result = self.docker.run(
                "network",
                "ls",
                "-q",
                "--filter",
                f"label=custback.phase6.run={self.run_id}",
                check=False,
            )
            networks = tuple(result.stdout.split())
        except Exception:
            pass
        for network in networks:
            self.docker.best_effort("network", "rm", network)

    def _assert_run_resources_removed(self) -> None:
        label = f"custback.phase6.run={self.run_id}"
        checks = (
            ("containers", ("ps", "-aq", "--filter", f"label={label}")),
            ("networks", ("network", "ls", "-q", "--filter", f"label={label}")),
            ("images", ("image", "ls", "-q", "--filter", f"label={label}")),
        )
        for kind, arguments in checks:
            result = self.docker.run(*arguments)
            remaining = tuple(result.stdout.split())
            if remaining:
                raise HarnessError(
                    f"two-host teardown left labeled {kind}: {remaining}"
                )

    def run(self) -> list[ScenarioResult]:
        base = self.temporary_base
        if base is not None:
            base = base.resolve(strict=True)
            if not base.is_dir():
                raise HarnessError("temporary base is not a directory")
        results: list[ScenarioResult] = []
        temporary = tempfile.TemporaryDirectory(
            prefix="custback-two-host-",
            dir=os.fspath(base) if base is not None else None,
        )
        try:
            run_root = Path(temporary.name)
            run_root.chmod(0o700)
            self._build_image(run_root)
            context = HarnessContext(
                self.docker,
                self.runner,
                run_root,
                self.run_id,
                self.image,
                MaterialFactory(self.runner),
                self.command_timeout,
                self.scenario_timeout,
            )
            for scenario_id in self.scenarios:
                started = time.monotonic()
                options = SCENARIO_OPTIONS.get(scenario_id, ScenarioOptions())
                with ScenarioStack(context, scenario_id, options) as stack:
                    SCENARIO_HANDLERS[scenario_id](stack)
                    results.append(
                        ScenarioResult(
                            scenario_id,
                            round(time.monotonic() - started, 3),
                            stack.meeting_address,
                            stack.renderer_address,
                        )
                    )
            if tuple(result.scenario_id for result in results) != self.scenarios:
                raise HarnessError("TLS scenario result set is incomplete or reordered")
            return results
        finally:
            try:
                self._sweep_run_resources()
            finally:
                try:
                    self.tracker.cleanup()
                finally:
                    try:
                        self._assert_run_resources_removed()
                    finally:
                        temporary.cleanup()


def _result_payload(
    harness: TwoHostHarness, results: Sequence[ScenarioResult]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "wheel_sha256": harness.wheel_sha256,
        "clean_host_evidence": harness.clean_host_evidence,
        "scenario_ids": [result.scenario_id for result in results],
        "scenarios": [
            {
                "id": result.scenario_id,
                "duration_s": result.duration_s,
                "meeting_address": result.meeting_address,
                "renderer_address": result.renderer_address,
            }
            for result in results
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="run packaged two-host WSS/HTTPS release qualification"
    )
    artifact_source = parser.add_mutually_exclusive_group(required=True)
    artifact_source.add_argument("--wheel", type=Path)
    artifact_source.add_argument("--artifacts", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scenario", dest="scenarios", action="append")
    parser.add_argument("--clean-host-evidence", action="store_true")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--scenario-timeout", type=float, default=30.0)
    parser.add_argument("--temporary-base", type=Path)
    parser.add_argument("--result", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command_timeout <= 0 or args.scenario_timeout <= 0:
        print("two-host system test: timeouts must be positive", file=sys.stderr)
        return 2
    harness: TwoHostHarness | None = None
    previous: dict[signal.Signals, Any] = {}

    def cancel(_signum, _frame) -> None:
        raise HarnessCancelled()

    if threading_main_process():
        for watched in (signal.SIGINT, signal.SIGTERM):
            previous[watched] = signal.signal(watched, cancel)
    try:
        wheel = args.wheel or wheel_from_artifacts(args.artifacts)
        harness = TwoHostHarness(
            wheel,
            manifest=args.manifest,
            docker_binary=args.docker,
            command_timeout=args.command_timeout,
            scenario_timeout=args.scenario_timeout,
            temporary_base=args.temporary_base,
            scenario_ids=args.scenarios,
            clean_host_evidence=args.clean_host_evidence,
        )
        results = harness.run()
        payload = _result_payload(harness, results)
        if args.result is not None:
            destination = args.result.resolve()
            if destination.exists() or destination.is_symlink():
                raise HarnessError("result destination already exists")
            _private_write(destination, (json.dumps(payload, indent=2) + "\n").encode())
        print(json.dumps(payload, sort_keys=True))
        return 0
    except HarnessCancelled:
        print("two-host system test: cancelled after teardown", file=sys.stderr)
        return 130
    except (HarnessError, OSError) as exc:
        print(f"two-host system test: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if harness is not None:
                try:
                    harness._sweep_run_resources()
                finally:
                    harness.tracker.cleanup()
        finally:
            for watched, handler in previous.items():
                signal.signal(watched, handler)


def threading_main_process() -> bool:
    """Keep signal registration mockable without importing threading at startup."""

    import threading

    return threading.current_thread() is threading.main_thread()


if __name__ == "__main__":
    raise SystemExit(main())
