"""RVM acceleration policy: provider selection, proof, and latched fallback.

ONNX Runtime can register a GPU execution provider and then silently create a
CPU-only session, or accept a session and then fail at inference time on a
missing DLL or an out-of-memory GPU.  This module turns the
:class:`~custback.config.AccelerationConfig` policy into concrete provider
choices, *proves* that a preferred provider can actually execute the Robust
Video Matting graph (not merely that it is registered), and models the GPU
lifecycle as a latched state machine so status reporting can be truthful.

Nothing here imports ``onnxruntime`` at module load; the segmenter passes in the
already-imported module.  Provider availability is a runtime outcome, never a
configuration error: an unavailable or unprovable GPU is resolved to CPU (in
``auto``) or to a clean startup failure (in ``gpu_required``).

Design companions: ``gpu_probe.py`` proves a tiny ``Add`` graph for diagnostics;
this module proves the *actual* RVM graph, which is the product evidence.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import sys
import threading
import time
import uuid as uuidlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .config import AccelerationConfig

log = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"

# Requested GPU provider name -> ORT execution provider name.
_PROVIDER_NAMES: dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": CPU_PROVIDER,
}
# Public-facing short label for an ORT provider name, used in status/logs.
_SHORT_LABELS: dict[str, str] = {
    "CUDAExecutionProvider": "cuda",
    "DmlExecutionProvider": "directml",
    "CoreMLExecutionProvider": "coreml",
    CPU_PROVIDER: "cpu",
}
# Preference order for ``provider: auto`` GPU discovery, most specific first.
_AUTO_GPU_ORDER = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
)


def provider_label(ort_provider: str) -> str:
    """Return the short device label ("cuda"/"directml"/"cpu") for a provider."""

    return _SHORT_LABELS.get(ort_provider, "cpu")


class AccelState(str, Enum):
    """Latched GPU lifecycle.  ``CPU_FALLBACK`` is terminal for a session."""

    STARTING = "starting"
    GPU_PROBING = "gpu_probing"
    GPU_ACTIVE = "gpu_active"
    CPU_FALLBACK = "cpu_fallback"


class GpuRequiredError(RuntimeError):
    """``mode: gpu_required`` could not prove accelerator execution."""


_CUDA_IDENTITY_SOURCE = "custback-cuda-driver-runtime-nvml-v1"
_CUDA_IDENTITY_ERROR = "CUDA device identity is unavailable"
_PCI_BUS_ID = re.compile(
    r"(?P<domain>[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):"
    r"(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])\Z"
)
_VERSION_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+){1,7}\Z")
_CUDA_COMPUTE_CAPABILITY_MAJOR = 75
_CUDA_COMPUTE_CAPABILITY_MINOR = 76
_NVML_ERROR_NOT_FOUND = 6
_MAX_CUDA_DEVICE_COUNT = 1024
_MAX_DEVICE_MEMORY_BYTES = 1 << 60
_MAX_CUDA_VERSION = 1_000_000


class _CudaUuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _load_cuda_identity_libraries() -> tuple[Any, Any, Any]:
    """Load platform CUDA/NVML ABIs without invoking external commands."""

    if sys.platform == "win32":
        cuda_loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
        return (
            cuda_loader("nvcuda.dll"),
            cuda_loader("cudart64_12.dll"),
            ctypes.CDLL("nvml.dll"),
        )
    elif sys.platform.startswith("linux"):
        return (
            ctypes.CDLL("libcuda.so.1"),
            ctypes.CDLL("libcudart.so.12"),
            ctypes.CDLL("libnvidia-ml.so.1"),
        )
    raise RuntimeError(_CUDA_IDENTITY_ERROR)


def _configure_cuda_identity_apis(driver: Any, runtime: Any, nvml: Any) -> None:
    """Declare only the stable C ABI surface consumed by the identity probe."""

    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceGetName.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetName.restype = ctypes.c_int
    uuid_getter = getattr(driver, "cuDeviceGetUuid_v2", None)
    if uuid_getter is None:
        uuid_getter = driver.cuDeviceGetUuid
    uuid_getter.argtypes = [ctypes.POINTER(_CudaUuid), ctypes.c_int]
    uuid_getter.restype = ctypes.c_int
    memory_getter = getattr(driver, "cuDeviceTotalMem_v2", None)
    if memory_getter is None:
        memory_getter = driver.cuDeviceTotalMem
    memory_getter.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    memory_getter.restype = ctypes.c_int
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetAttribute.restype = ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int

    runtime.cudaDeviceGetPCIBusId.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.c_int,
    ]
    runtime.cudaDeviceGetPCIBusId.restype = ctypes.c_int
    runtime.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaDriverGetVersion.restype = ctypes.c_int
    runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaRuntimeGetVersion.restype = ctypes.c_int

    init = getattr(nvml, "nvmlInit_v2", None)
    if init is None:
        init = nvml.nvmlInit
    init.argtypes = []
    init.restype = ctypes.c_int
    nvml.nvmlShutdown.argtypes = []
    nvml.nvmlShutdown.restype = ctypes.c_int
    nvml.nvmlSystemGetDriverVersion.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint,
    ]
    nvml.nvmlSystemGetDriverVersion.restype = ctypes.c_int
    nvml.nvmlSystemGetNVMLVersion.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint,
    ]
    nvml.nvmlSystemGetNVMLVersion.restype = ctypes.c_int
    nvml.nvmlDeviceGetHandleByUUID.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    nvml.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
    nvml.nvmlDeviceGetUUID.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint,
    ]
    nvml.nvmlDeviceGetUUID.restype = ctypes.c_int
    nvml.nvmlDeviceGetName.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint,
    ]
    nvml.nvmlDeviceGetName.restype = ctypes.c_int
    nvml.nvmlDeviceGetMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NvmlMemory),
    ]
    nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int


def _require_identity_success(result: object) -> None:
    if type(result) is not int or result != 0:
        raise RuntimeError(_CUDA_IDENTITY_ERROR)


def _identity_text(buffer: Any, *, maximum: int) -> str:
    try:
        value = bytes(buffer.value).decode("ascii")
    except (AttributeError, UnicodeError):
        raise RuntimeError(_CUDA_IDENTITY_ERROR) from None
    if (
        not value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(_CUDA_IDENTITY_ERROR)
    return value


def _identity_version_text(buffer: Any, *, maximum: int) -> str:
    value = _identity_text(buffer, maximum=maximum)
    if _VERSION_TEXT.fullmatch(value) is None or not any(
        int(part) > 0 for part in value.split(".")
    ):
        raise RuntimeError(_CUDA_IDENTITY_ERROR)
    return value


def _positive_identity_int(value: int, *, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise RuntimeError(_CUDA_IDENTITY_ERROR)
    return value


def _canonical_pci_bus_id(value: str) -> str:
    matched = _PCI_BUS_ID.fullmatch(value)
    if matched is None:
        raise RuntimeError(_CUDA_IDENTITY_ERROR)
    return (
        f"{int(matched['domain'], 16):08x}:"
        f"{int(matched['bus'], 16):02x}:"
        f"{int(matched['device'], 16):02x}."
        f"{int(matched['function'])}"
    )


def _nvml_identity(
    nvml: Any,
    *,
    raw_uuid: bytes,
) -> tuple[str, str, ctypes.c_void_p]:
    canonical = str(uuidlib.UUID(bytes=raw_uuid))
    matches: list[tuple[str, str, ctypes.c_void_p]] = []
    for kind, prefix in (("gpu", "GPU-"), ("mig", "MIG-")):
        candidate = f"{prefix}{canonical}"
        handle = ctypes.c_void_p()
        result = nvml.nvmlDeviceGetHandleByUUID(
            candidate.encode("ascii"),
            ctypes.byref(handle),
        )
        if type(result) is not int:
            raise RuntimeError(_CUDA_IDENTITY_ERROR)
        if result == _NVML_ERROR_NOT_FOUND:
            continue
        if result != 0 or handle.value is None:
            raise RuntimeError(_CUDA_IDENTITY_ERROR)
        matches.append((candidate, kind, handle))
    if len(matches) != 1:
        raise RuntimeError(_CUDA_IDENTITY_ERROR)
    return matches[0]


def _collect_cuda_device_identity(device_id: int) -> dict[str, object]:
    driver, runtime, nvml = _load_cuda_identity_libraries()
    _configure_cuda_identity_apis(driver, runtime, nvml)

    _require_identity_success(driver.cuInit(0))
    device_count = ctypes.c_int()
    _require_identity_success(driver.cuDeviceGetCount(ctypes.byref(device_count)))
    if (
        not 0 < device_count.value <= _MAX_CUDA_DEVICE_COUNT
        or device_id >= device_count.value
    ):
        raise RuntimeError(_CUDA_IDENTITY_ERROR)
    device = ctypes.c_int()
    _require_identity_success(driver.cuDeviceGet(ctypes.byref(device), device_id))

    cuda_name_buffer = ctypes.create_string_buffer(256)
    _require_identity_success(
        driver.cuDeviceGetName(cuda_name_buffer, len(cuda_name_buffer), device.value)
    )
    cuda_name = _identity_text(cuda_name_buffer, maximum=255)

    raw_uuid = _CudaUuid()
    uuid_getter = getattr(driver, "cuDeviceGetUuid_v2", None)
    if uuid_getter is None:
        uuid_getter = driver.cuDeviceGetUuid
    _require_identity_success(uuid_getter(ctypes.byref(raw_uuid), device.value))
    uuid_bytes = bytes(raw_uuid.bytes)
    if len(uuid_bytes) != 16 or not any(uuid_bytes):
        raise RuntimeError(_CUDA_IDENTITY_ERROR)

    cuda_memory = ctypes.c_size_t()
    memory_getter = getattr(driver, "cuDeviceTotalMem_v2", None)
    if memory_getter is None:
        memory_getter = driver.cuDeviceTotalMem
    _require_identity_success(memory_getter(ctypes.byref(cuda_memory), device.value))
    cuda_memory_bytes = _positive_identity_int(
        int(cuda_memory.value),
        maximum=_MAX_DEVICE_MEMORY_BYTES,
    )

    compute_major = ctypes.c_int()
    compute_minor = ctypes.c_int()
    _require_identity_success(
        driver.cuDeviceGetAttribute(
            ctypes.byref(compute_major),
            _CUDA_COMPUTE_CAPABILITY_MAJOR,
            device.value,
        )
    )
    _require_identity_success(
        driver.cuDeviceGetAttribute(
            ctypes.byref(compute_minor),
            _CUDA_COMPUTE_CAPABILITY_MINOR,
            device.value,
        )
    )
    major = _positive_identity_int(compute_major.value, maximum=99)
    if not 0 <= compute_minor.value <= 99:
        raise RuntimeError(_CUDA_IDENTITY_ERROR)
    minor = int(compute_minor.value)

    driver_version = ctypes.c_int()
    runtime_driver_version = ctypes.c_int()
    runtime_version = ctypes.c_int()
    _require_identity_success(driver.cuDriverGetVersion(ctypes.byref(driver_version)))
    _require_identity_success(
        runtime.cudaDriverGetVersion(ctypes.byref(runtime_driver_version))
    )
    _require_identity_success(
        runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
    )
    cuda_driver_version = _positive_identity_int(
        driver_version.value,
        maximum=_MAX_CUDA_VERSION,
    )
    if runtime_driver_version.value != cuda_driver_version:
        raise RuntimeError(_CUDA_IDENTITY_ERROR)
    cuda_runtime_version = _positive_identity_int(
        runtime_version.value,
        maximum=_MAX_CUDA_VERSION,
    )

    driver_pci_buffer = ctypes.create_string_buffer(32)
    _require_identity_success(
        driver.cuDeviceGetPCIBusId(
            driver_pci_buffer,
            len(driver_pci_buffer),
            device.value,
        )
    )
    driver_pci_bus_id = _canonical_pci_bus_id(
        _identity_text(driver_pci_buffer, maximum=31)
    )

    runtime_pci_buffer = ctypes.create_string_buffer(32)
    _require_identity_success(
        runtime.cudaDeviceGetPCIBusId(
            runtime_pci_buffer,
            len(runtime_pci_buffer),
            device_id,
        )
    )
    pci_bus_id = _canonical_pci_bus_id(_identity_text(runtime_pci_buffer, maximum=31))
    if driver_pci_bus_id != pci_bus_id:
        raise RuntimeError(_CUDA_IDENTITY_ERROR)

    nvml_initialized = False
    try:
        init = getattr(nvml, "nvmlInit_v2", None)
        if init is None:
            init = nvml.nvmlInit
        _require_identity_success(init())
        nvml_initialized = True

        uuid_value, uuid_kind, nvml_handle = _nvml_identity(
            nvml,
            raw_uuid=uuid_bytes,
        )
        nvml_uuid_buffer = ctypes.create_string_buffer(96)
        _require_identity_success(
            nvml.nvmlDeviceGetUUID(
                nvml_handle,
                nvml_uuid_buffer,
                len(nvml_uuid_buffer),
            )
        )
        if _identity_text(nvml_uuid_buffer, maximum=95) != uuid_value:
            raise RuntimeError(_CUDA_IDENTITY_ERROR)

        nvml_name_buffer = ctypes.create_string_buffer(256)
        _require_identity_success(
            nvml.nvmlDeviceGetName(
                nvml_handle,
                nvml_name_buffer,
                len(nvml_name_buffer),
            )
        )
        nvml_name = _identity_text(nvml_name_buffer, maximum=255)

        nvml_memory = _NvmlMemory()
        _require_identity_success(
            nvml.nvmlDeviceGetMemoryInfo(
                nvml_handle,
                ctypes.byref(nvml_memory),
            )
        )
        nvml_memory_bytes = _positive_identity_int(
            int(nvml_memory.total),
            maximum=_MAX_DEVICE_MEMORY_BYTES,
        )

        nvidia_driver_buffer = ctypes.create_string_buffer(128)
        _require_identity_success(
            nvml.nvmlSystemGetDriverVersion(
                nvidia_driver_buffer,
                len(nvidia_driver_buffer),
            )
        )
        nvidia_driver_version = _identity_version_text(
            nvidia_driver_buffer,
            maximum=127,
        )

        nvml_version_buffer = ctypes.create_string_buffer(128)
        _require_identity_success(
            nvml.nvmlSystemGetNVMLVersion(
                nvml_version_buffer,
                len(nvml_version_buffer),
            )
        )
        nvml_version = _identity_version_text(nvml_version_buffer, maximum=127)
    finally:
        if nvml_initialized:
            _require_identity_success(nvml.nvmlShutdown())

    return {
        "identity_source": _CUDA_IDENTITY_SOURCE,
        "ordinal": device_id,
        "uuid": uuid_value,
        "uuid_kind": uuid_kind,
        "pci_bus_id": pci_bus_id,
        "cuda_name": cuda_name,
        "nvml_name": nvml_name,
        "cuda_total_memory_bytes": cuda_memory_bytes,
        "nvml_total_memory_bytes": nvml_memory_bytes,
        "compute_capability_major": major,
        "compute_capability_minor": minor,
        "cuda_driver_version": cuda_driver_version,
        "cuda_runtime_version": cuda_runtime_version,
        "nvidia_driver_version": nvidia_driver_version,
        "nvml_version": nvml_version,
    }


def collect_cuda_device_identity(device_id: int) -> dict[str, object]:
    """Return exact, content-free-bindable identity for one visible CUDA device.

    The returned values contain no paths, hostnames, pixels, or mutable
    utilization state. Callers should retain only a digest of this object when
    publishing content-free evidence.
    """

    if type(device_id) is not int or not 0 <= device_id <= 64:
        raise ValueError("CUDA device id must be an integer in [0, 64]")
    try:
        return _collect_cuda_device_identity(device_id)
    except Exception:
        raise GpuRequiredError(_CUDA_IDENTITY_ERROR) from None


@dataclass(frozen=True)
class ProviderCandidate:
    """One provider ORT should be asked to use, with its device options."""

    name: str
    options: dict[str, Any]

    def as_ort_arg(self) -> Any:
        """Return the provider in ORT's ``providers=`` list form."""

        return (self.name, self.options) if self.options else self.name


@dataclass(frozen=True)
class AccelerationStatus:
    """Truthful, JSON-safe snapshot of the acceleration lifecycle.

    ``active_provider`` reflects the provider the session is *actually* running
    on after any fallback, never the requested or inferred order.
    """

    requested_mode: str
    requested_provider: str
    device_id: int
    state: str
    active_provider: str
    fallback_active: bool
    fallback_reason: str
    fallback_count: int
    last_transition_ms: float | None

    def public_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "requested_provider": self.requested_provider,
            "device_id": self.device_id,
            "state": self.state,
            "active_provider": self.active_provider,
            "fallback_active": self.fallback_active,
            "fallback_reason": self.fallback_reason,
            "fallback_count": self.fallback_count,
            "last_transition_ms": self.last_transition_ms,
        }


def _bounded_reason(text: str) -> str:
    """Collapse a failure into a bounded, credential/path-free status reason."""

    words = str(text).split()
    reason = " ".join(words)[:200]
    return reason or "unknown"


class AccelerationState:
    """Thread-safe latched acceleration lifecycle with truthful status.

    The RVM segmenter constructs one of these and mutates it from both the
    setup thread (startup probe) and the pipeline worker (inference-time
    recovery); the API/status thread only reads.  Once latched to
    ``CPU_FALLBACK`` the state never returns to GPU for the session's life.
    """

    def __init__(self, cfg: AccelerationConfig):
        self._lock = threading.Lock()
        self._requested_mode = cfg.mode
        self._requested_provider = cfg.provider
        self._device_id = cfg.device_id
        self._state = AccelState.STARTING
        self._active_provider = CPU_PROVIDER
        self._fallback_active = False
        self._fallback_reason = ""
        self._fallback_count = 0
        self._last_transition = time.monotonic()

    def _touch(self) -> None:
        self._last_transition = time.monotonic()

    def mark_probing(self) -> None:
        with self._lock:
            if self._state is AccelState.CPU_FALLBACK:
                return
            self._state = AccelState.GPU_PROBING
            self._touch()

    def mark_gpu_active(self, ort_provider: str) -> None:
        with self._lock:
            if self._state is AccelState.CPU_FALLBACK:
                return
            self._state = AccelState.GPU_ACTIVE
            self._active_provider = ort_provider
            self._fallback_active = False
            self._fallback_reason = ""
            self._touch()

    def mark_cpu_active(self) -> None:
        """Settle on CPU as the intended provider without flagging a fallback.

        Used when the policy never attempted a GPU (``mode: cpu``, or ``auto``
        with no GPU provider registered at all), so CPU is the correct outcome
        rather than a degraded one.  ``CPU_FALLBACK`` is the single on-CPU
        terminal state; ``fallback_active`` distinguishes this intended landing
        from a degraded one, and no fallback is counted.
        """

        with self._lock:
            if self._state is AccelState.CPU_FALLBACK:
                return
            self._state = AccelState.CPU_FALLBACK
            self._active_provider = CPU_PROVIDER
            self._fallback_active = False
            self._fallback_reason = ""
            self._touch()

    def latch_fallback(self, reason: str) -> None:
        """Latch to CPU after a GPU probe or inference-time GPU failure."""

        with self._lock:
            first = not self._fallback_active
            self._state = AccelState.CPU_FALLBACK
            self._active_provider = CPU_PROVIDER
            self._fallback_active = True
            self._fallback_reason = _bounded_reason(reason)
            if first:
                self._fallback_count += 1
            self._touch()

    @property
    def fallback_latched(self) -> bool:
        with self._lock:
            return self._state is AccelState.CPU_FALLBACK

    @property
    def on_gpu(self) -> bool:
        """True while inference is running on a GPU execution provider."""

        with self._lock:
            return (
                self._state is AccelState.GPU_ACTIVE
                and self._active_provider != CPU_PROVIDER
            )

    def status(self) -> AccelerationStatus:
        with self._lock:
            age_ms = (time.monotonic() - self._last_transition) * 1000.0
            return AccelerationStatus(
                requested_mode=self._requested_mode,
                requested_provider=self._requested_provider,
                device_id=self._device_id,
                state=self._state.value,
                active_provider=provider_label(self._active_provider),
                fallback_active=self._fallback_active,
                fallback_reason=self._fallback_reason,
                fallback_count=self._fallback_count,
                last_transition_ms=round(age_ms, 1),
            )


def resolve_provider_candidates(
    cfg: AccelerationConfig,
    available: list[str],
) -> list[ProviderCandidate]:
    """Return the ordered GPU providers to attempt for this policy.

    The CPU provider is always appended by the caller as the final safety net;
    this returns only the *preferred* (GPU) candidates, which is an empty list
    for ``mode: cpu`` or when no matching provider is registered.
    """

    if cfg.mode == "cpu":
        return []
    available_set = set(available)
    if cfg.provider == "auto":
        wanted = [name for name in _AUTO_GPU_ORDER if name in available_set]
    else:
        name = _PROVIDER_NAMES[cfg.provider]
        wanted = [name] if name in available_set else []
    candidates: list[ProviderCandidate] = []
    for name in wanted:
        candidates.append(
            ProviderCandidate(name, _provider_options(name, cfg.device_id))
        )
    return candidates


def _provider_options(name: str, device_id: int) -> dict[str, Any]:
    if name == "CUDAExecutionProvider":
        return {"device_id": device_id}
    if name == "DmlExecutionProvider":
        return {"device_id": device_id}
    return {}


def preload_acceleration_dlls(provider: str) -> None:
    """Best-effort DLL directory preload for a Windows/CUDA ORT provider.

    On Windows, onnxruntime-gpu resolves CUDA/cuDNN DLLs from the process DLL
    search path.  A frozen build or a fresh install may not have the CUDA
    ``bin`` directory on ``PATH``; adding common locations to the DLL search
    path *before* the first session is constructed avoids a provider that is
    "registered" but fails to initialize.  This is intentionally non-fatal:
    failure here simply means the probe below decides GPU is unavailable.
    """

    if sys.platform != "win32" or provider != "CUDAExecutionProvider":
        return
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:  # pragma: no cover - Windows Python always has it
        return
    seen: set[str] = set()
    for raw in _candidate_cuda_dll_dirs():
        try:
            resolved = Path(raw)
            if not resolved.is_dir():
                continue
            key = os.path.normcase(str(resolved))
            if key in seen:
                continue
            seen.add(key)
            add_dll_directory(str(resolved))
            log.debug("added CUDA DLL directory to search path: %s", resolved)
        except OSError:
            continue


def _candidate_cuda_dll_dirs() -> list[str]:  # pragma: no cover - Windows-only paths
    dirs: list[str] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dirs.append(str(Path(cuda_path) / "bin"))
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip()
        if entry and ("cuda" in entry.lower() or "cudnn" in entry.lower()):
            dirs.append(entry)
    return dirs


def synthetic_rvm_feeds(height: int = 64, width: int = 64) -> dict[str, np.ndarray]:
    """Build one valid set of RVM inputs for a warm-up / proof inference."""

    src = np.zeros((1, 3, height, width), dtype=np.float32)
    # A faint gradient exercises real matting nodes rather than an all-zero
    # short-circuit, without depending on any camera-derived content.
    src[:, :, : height // 2, :] = 0.6
    zero = np.zeros((1, 1, 1, 1), dtype=np.float32)
    return {
        "src": src,
        "r1i": zero,
        "r2i": zero,
        "r3i": zero,
        "r4i": zero,
        "downsample_ratio": np.asarray([0.25], dtype=np.float32),
    }


def _profile_provider_executed(events: Any, provider: str) -> bool:
    """Return True if the profile shows a real node ran on ``provider``."""

    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        args = event.get("args")
        if isinstance(args, dict) and args.get("provider") == provider:
            return True
    return False


@dataclass(frozen=True)
class ProofResult:
    """Outcome of a real-RVM provider proof."""

    proven: bool
    active_providers: tuple[str, ...]
    error: str = ""


def prove_rvm_provider(
    ort: Any,
    model_path: str | bytes,
    candidate: ProviderCandidate,
    *,
    height: int = 64,
    width: int = 64,
) -> ProofResult:
    """Prove ``candidate`` can execute the actual RVM graph, via profiling.

    A short-lived profiling session runs one synthetic RVM frame and the ORT
    profile is inspected for a node executed on the candidate provider.  This
    is discarded afterwards; the production session is built separately without
    profiling overhead.  The proof mirrors the production provider order,
    including its explicit CPU safety provider.  Registration-without-execution
    and initialization fallback are still caught because the evidence is a real
    executed node, not merely the provider list.
    """

    session = None
    profile_path = ""
    try:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.log_severity_level = 3
        # Do not set session.disable_cpu_ep_fallback here.  The proof explicitly
        # registers CPUExecutionProvider to match the production session, and
        # newer ONNX Runtime releases reject that provider list when CPU fallback
        # is simultaneously disabled.  The profile below is the authoritative
        # guard against a registered CUDA/DirectML provider that executes no RVM
        # nodes.
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=[candidate.as_ort_arg(), CPU_PROVIDER],
        )
        active = tuple(session.get_providers())
        if candidate.name not in active:
            return ProofResult(False, active, "provider not active in session")
        session.run(None, synthetic_rvm_feeds(height, width))
        profile_path = session.end_profiling()
        session = None
        proven = False
        if profile_path:
            try:
                with Path(profile_path).open(encoding="utf-8") as stream:
                    proven = _profile_provider_executed(
                        json.load(stream), candidate.name
                    )
            except (OSError, ValueError):
                proven = False
        if not proven:
            return ProofResult(False, active, "no node executed on the GPU provider")
        return ProofResult(True, active, "")
    except Exception as exc:
        return ProofResult(False, (), _bounded_reason(str(exc)))
    finally:
        if session is not None:
            try:
                session.end_profiling()
            except Exception:
                pass
        if profile_path:
            try:
                Path(profile_path).unlink()
            except OSError:
                pass


def warm_up_session(session: Any, height: int, width: int) -> None:
    """Run one synthetic frame so first real frame pays no cold-start cost.

    Recurrent-state shape is decoupled from the warm-up frame, so this never
    perturbs the segmenter's own recurrent state (the segmenter resets on the
    first real frame anyway when the resolution differs).
    """

    session.run(None, synthetic_rvm_feeds(height, width))
