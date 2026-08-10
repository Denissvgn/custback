"""Offline MATTE-5.3 performance and platform qualification authority.

The qualifier joins already-recorded, owner-only evidence.  It deliberately
opens no camera, model, preview, API, network service, or output sink.  Local
physical origin remains an operator attestation; generated evidence can test
the contract but can never qualify a production profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, cast

from .compositor import COMPOSITOR_SUBSTAGE_NAMES
from .capture_diagnostics import (
    REPORT_SCHEMA as CAPTURE_REPORT_SCHEMA,
    REPORT_VERSION as CAPTURE_REPORT_VERSION,
)
from .matte_diagnostics import (
    MatteDiagnosticsError,
    _atomic_private_write,
    _json_bytes,
    _private_directory,
    _read_private_file,
)
from .matte_performance import (
    REPORT_SCHEMA as PERFORMANCE_REPORT_SCHEMA,
    REPORT_VERSION as PERFORMANCE_REPORT_VERSION,
    MattePerformanceError,
    validate_performance_report,
)
from .matte_rvm_qualification import (
    REPORT_SCHEMA as RVM_REPORT_SCHEMA,
    REPORT_VERSION as RVM_REPORT_VERSION,
)
from .matte_visual_qualification import (
    REPORT_SCHEMA as VISUAL_REPORT_SCHEMA,
    REPORT_VERSION as VISUAL_REPORT_VERSION,
)

PLAN_SCHEMA = "custback.matte-platform-qualification-plan"
PLAN_VERSION = 1
RUN_SCHEMA = "custback.matte-platform-run-evidence"
RUN_VERSION = 1
REPORT_SCHEMA = "custback.matte-platform-qualification-report"
REPORT_VERSION = 1

MIN_WARMUP_FRAMES = 30
MIN_MEASURED_FRAMES = 300
MIN_MEASURED_SECONDS = 10.0
MIN_SOAK_SECONDS = 1_800.0
MIN_RESOURCE_SAMPLES = 31
MAX_RESOURCE_SAMPLE_GAP_S = 60.0
MIN_UNIQUE_COMPOSITES_PER_S = 27.0
MIN_CAPTURE_AND_OUTPUT_TARGET_RATIO = 0.90
MAX_COMPLETE_SERVICE_P95_MS = 33.333334
MAX_COMPOSITOR_P95_720P_MS = 22.0
MAX_REFINEMENT_P95_720P_MS = 5.0
MAX_SHUTDOWN_MS = 5_000.0
MAX_RSS_DRIFT_BYTES = 64 * 1024 * 1024
MAX_VRAM_DRIFT_BYTES = 64 * 1024 * 1024
MAX_RSS_SPAN_BYTES = 256 * 1024 * 1024
MAX_VRAM_SPAN_BYTES = 256 * 1024 * 1024
MAX_E2E_AGE_DRIFT_MS = 33.333334

ROUTE_CONTRACTS: dict[str, dict[str, str]] = {
    "linux-v4l2-pyvirtualcam": {
        "platform": "linux",
        "capture_backend": "V4L2",
        "sink_backend": "pyvirtualcam",
        "consumer": "v4l2loopback",
    },
    "macos-obsvcam-pyvirtualcam": {
        "platform": "darwin",
        "capture_backend": "AVFOUNDATION",
        "sink_backend": "pyvirtualcam",
        "consumer": "obs-virtual-camera",
    },
    "windows-msmf-pyvirtualcam": {
        "platform": "windows",
        "capture_backend": "MSMF",
        "sink_backend": "pyvirtualcam",
        "consumer": "obs-virtual-camera",
    },
    "windows-dshow-pyvirtualcam": {
        "platform": "windows",
        "capture_backend": "DSHOW",
        "sink_backend": "pyvirtualcam",
        "consumer": "obs-virtual-camera",
    },
    "windows-native": {
        "platform": "windows",
        "capture_backend": "MSMF",
        "sink_backend": "native",
        "consumer": "media-foundation-virtual-camera",
    },
}

PROFILE_CONTRACTS: dict[str, dict[str, object]] = {
    "rvm_matting": {
        "backend": "rvm",
        "segmenter": "RVMSegmenter",
        "backend_kind": "matting",
        "quality_claim": True,
    },
    "mediapipe_segmentation": {
        "backend": "mediapipe",
        "segmenter": "MediaPipeSegmenter",
        "backend_kind": "segmentation",
        "quality_claim": True,
    },
    "heuristic_segmentation": {
        "backend": "heuristic",
        "segmenter": "HeuristicSegmenter",
        "backend_kind": "heuristic",
        "quality_claim": False,
    },
    "none_passthrough": {
        "backend": "none",
        "segmenter": "NullSegmenter",
        "backend_kind": "none",
        "quality_claim": False,
    },
}

PROVIDERS = ("cpu", "cuda", "directml")
PROVIDER_RUNTIME_NAMES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}
DEPENDENCY_PROFILES = (
    "standard",
    "without_mediapipe",
    "without_gpu_provider",
)
CANVASES = ((640, 360), (1280, 720), (1920, 1080))
TARGET_FPS = 30

MAX_PLAN_BYTES = 4 * 1024 * 1024
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_RUN_BYTES = 64 * 1024 * 1024
MAX_CELLS = 256
MAX_SAMPLES = 250_000

_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+()-]{0,255}$")


class MattePlatformQualificationError(ValueError):
    """A MATTE-5.3 plan or evidence document is unsafe or inconsistent."""


def _strict_mapping(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise MattePlatformQualificationError(
            f"{name} must contain exactly {sorted(keys)}"
        )
    return dict(value)


def _strict_sequence(value: object, name: str, *, maximum: int) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or len(value) > maximum
    ):
        raise MattePlatformQualificationError(f"{name} is invalid")
    return list(value)


def _safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise MattePlatformQualificationError(f"{name} is invalid")
    return value


def _safe_text(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_TEXT.fullmatch(value) is None:
        raise MattePlatformQualificationError(f"{name} is invalid")
    return value


def _platform_family(value: object) -> str | None:
    """Normalize an upstream platform label without weakening route identity."""

    if not isinstance(value, str):
        return None
    normalized = value.lower()
    if normalized == "linux" or normalized.startswith("linux-"):
        return "linux"
    if normalized in ("darwin", "macos") or normalized.startswith(
        ("darwin-", "macos-")
    ):
        return "darwin"
    if normalized in ("windows", "win32") or normalized.startswith(
        ("windows-", "win32-")
    ):
        return "windows"
    return None


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MattePlatformQualificationError(f"{name} is invalid")
    return value


def _exact_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise MattePlatformQualificationError(f"{name} is invalid")
    return value


def _finite(
    value: object,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0e12,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise MattePlatformQualificationError(f"{name} is invalid")
    return float(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _report_digest(report: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in report.items() if key != "evidence_sha256"}
    return _sha256_bytes(_json_bytes(unsigned))


def _canonical_digest(value: object) -> str:
    """Return the digest used to bind one normalized, path-free value."""

    return _sha256_bytes(_json_bytes(value))


def _relative_path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str):
        raise MattePlatformQualificationError(f"{name} must be a relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in value
    ):
        raise MattePlatformQualificationError(f"{name} must be a safe relative path")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise MattePlatformQualificationError(
            f"{name} owner-only root is unavailable"
        ) from exc
    candidate = resolved_root.joinpath(*pure.parts)
    current = resolved_root
    for part in pure.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            # Content-free templates may name an evidence directory that has
            # not been collected yet.  Qualification will fail closed when the
            # artifact is opened; any component that does exist now must be a
            # real owner-only directory rather than a symlink escape.
            break
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MattePlatformQualificationError(
                f"{name} must not traverse a symlink or non-directory"
            )
        try:
            _private_directory(current, create=False)
        except (MatteDiagnosticsError, OSError) as exc:
            raise MattePlatformQualificationError(
                f"{name} intermediate directories must be owner-only"
            ) from exc
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except FileNotFoundError:
        return candidate
    except OSError as exc:
        raise MattePlatformQualificationError(f"{name} is unavailable") from exc
    if resolved_candidate.parent != resolved_root and resolved_root not in (
        resolved_candidate.parent,
        *resolved_candidate.parents,
    ):
        raise MattePlatformQualificationError(
            f"{name} must remain below the qualification-plan directory"
        )
    if resolved_candidate != candidate:
        raise MattePlatformQualificationError(f"{name} must not traverse a symlink")
    return candidate


def _load_json_file(
    path: Path, *, expected_sha256: str, maximum: int
) -> dict[str, Any]:
    try:
        payload = _read_private_file(path, max_bytes=maximum)
    except MatteDiagnosticsError as exc:
        raise MattePlatformQualificationError(str(exc)) from exc
    if _sha256_bytes(payload) != expected_sha256:
        raise MattePlatformQualificationError("private evidence digest does not match")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MattePlatformQualificationError(
            "private evidence is not valid JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise MattePlatformQualificationError("private evidence root must be an object")
    return dict(value)


def _descriptor(value: object, root: Path, name: str) -> dict[str, object]:
    item = _strict_mapping(value, {"path", "sha256", "evidence_sha256"}, name)
    path = _relative_path(root, item["path"], f"{name}.path")
    return {
        "path": path,
        "path_token": cast(str, item["path"]),
        "sha256": _digest(item["sha256"], f"{name}.sha256"),
        "evidence_sha256": _digest(item["evidence_sha256"], f"{name}.evidence_sha256"),
    }


def _file_descriptor(value: object, root: Path, name: str) -> dict[str, object]:
    item = _strict_mapping(value, {"path", "sha256"}, name)
    path = _relative_path(root, item["path"], f"{name}.path")
    return {
        "path": path,
        "path_token": cast(str, item["path"]),
        "sha256": _digest(item["sha256"], f"{name}.sha256"),
    }


def _load_plan(plan_path: Path | str) -> tuple[dict[str, Any], Path, str]:
    """Load and normalize one owner-only strict v1 plan."""

    path = Path(plan_path)
    try:
        payload = _read_private_file(path, max_bytes=MAX_PLAN_BYTES)
    except MatteDiagnosticsError as exc:
        raise MattePlatformQualificationError(str(exc)) from exc
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MattePlatformQualificationError(
            "qualification plan is not valid JSON"
        ) from exc
    root = _strict_mapping(
        raw,
        {
            "schema",
            "version",
            "qualification",
            "provenance",
            "reactions",
            "candidate",
            "prerequisites",
            "routes",
            "profiles",
            "cells",
        },
        "qualification plan",
    )
    if (
        root["schema"] != PLAN_SCHEMA
        or type(root["version"]) is not int
        or root["version"] != PLAN_VERSION
    ):
        raise MattePlatformQualificationError("unsupported qualification plan")

    qualification = _strict_mapping(root["qualification"], {"id"}, "qualification")
    qualification["id"] = _safe_id(qualification["id"], "qualification.id")

    provenance = _strict_mapping(
        root["provenance"],
        {
            "kind",
            "license_or_consent_sha256",
            "physical_capture_origin_attested",
            "physical_consumer_origin_attested",
            "same_host_attested",
            "physical_origin_cryptographically_proven",
        },
        "provenance",
    )
    if provenance["kind"] not in ("generated", "consented-local", "licensed-local"):
        raise MattePlatformQualificationError("provenance.kind is invalid")
    generated = provenance["kind"] == "generated"
    for key in (
        "physical_capture_origin_attested",
        "physical_consumer_origin_attested",
        "same_host_attested",
        "physical_origin_cryptographically_proven",
    ):
        if type(provenance[key]) is not bool:
            raise MattePlatformQualificationError(f"provenance.{key} must be boolean")
    if provenance["physical_origin_cryptographically_proven"] is not False:
        raise MattePlatformQualificationError(
            "physical origin cannot be represented as cryptographically proven"
        )
    if generated:
        if provenance["license_or_consent_sha256"] is not None or any(
            provenance[key]
            for key in (
                "physical_capture_origin_attested",
                "physical_consumer_origin_attested",
                "same_host_attested",
            )
        ):
            raise MattePlatformQualificationError(
                "generated provenance cannot claim local physical authority"
            )
    else:
        _digest(
            provenance["license_or_consent_sha256"],
            "provenance.license_or_consent_sha256",
        )

    reactions = _strict_mapping(
        root["reactions"], {"enabled", "post_base_event_count"}, "reactions"
    )
    if (
        reactions["enabled"] is not False
        or type(reactions["post_base_event_count"]) is not int
        or reactions["post_base_event_count"] != 0
    ):
        raise MattePlatformQualificationError(
            "MATTE-5.3 evidence must have reactions disabled"
        )

    candidate = _strict_mapping(
        root["candidate"], {"revision", "build_sha256", "defaults_changed"}, "candidate"
    )
    candidate["revision"] = _safe_id(candidate["revision"], "candidate.revision")
    candidate["build_sha256"] = _digest(
        candidate["build_sha256"], "candidate.build_sha256"
    )
    if candidate["defaults_changed"] is not False:
        raise MattePlatformQualificationError("MATTE-5.3 cannot change defaults")

    prerequisites = _strict_mapping(
        root["prerequisites"], {"visual", "performance", "rvm"}, "prerequisites"
    )
    try:
        plan_root = path.parent.resolve(strict=True)
        _private_directory(plan_root, create=False)
    except (MatteDiagnosticsError, OSError) as exc:
        raise MattePlatformQualificationError(
            "qualification plan directory must be owner-only"
        ) from exc
    normalized_prerequisites = {
        name: _descriptor(prerequisites[name], plan_root, f"prerequisites.{name}")
        for name in ("visual", "performance", "rvm")
    }

    expected_routes = [
        {"id": route_id, **contract} for route_id, contract in ROUTE_CONTRACTS.items()
    ]
    if root["routes"] != expected_routes:
        raise MattePlatformQualificationError(
            "route catalog must exactly match the reviewed platform routes"
        )

    profiles = _load_profiles(root["profiles"])
    cells = _load_cells(root["cells"], plan_root, profiles)
    _validate_matrix_coverage(profiles, cells)
    authority_descriptors = list(normalized_prerequisites.values()) + [
        cast(Mapping[str, object], cell["run"])
        for cell in cells
        if cell["state"] == "recorded"
    ]
    if len({item["path_token"] for item in authority_descriptors}) != len(
        authority_descriptors
    ) or len({item["sha256"] for item in authority_descriptors}) != len(
        authority_descriptors
    ):
        raise MattePlatformQualificationError(
            "prerequisite and recorded run artifacts must be distinct"
        )
    authority_paths = {cast(str, item["path_token"]) for item in authority_descriptors}
    authority_digests = {cast(str, item["sha256"]) for item in authority_descriptors}
    capture_groups: dict[tuple[str, str], set[tuple[object, ...]]] = {}
    for cell in cells:
        if cell["state"] != "recorded":
            continue
        capture = cast(Mapping[str, object], cell["capture_report"])
        if (
            capture["path_token"] in authority_paths
            or capture["sha256"] in authority_digests
        ):
            raise MattePlatformQualificationError(
                "capture reports must not alias prerequisite or run artifacts"
            )
        scope = (
            cell["route_id"],
            cast(Mapping[str, object], cell["canvas"])["width"],
            cast(Mapping[str, object], cell["canvas"])["height"],
            cell["target_fps"],
        )
        for kind, token in (
            ("path", cast(str, capture["path_token"])),
            ("sha256", cast(str, capture["sha256"])),
        ):
            capture_groups.setdefault((kind, token), set()).add(scope)
    if any(len(scopes) != 1 for scopes in capture_groups.values()):
        raise MattePlatformQualificationError(
            "a reused capture report must retain one route/canvas/FPS scope"
        )
    normalized = {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "qualification": qualification,
        "provenance": provenance,
        "reactions": reactions,
        "candidate": candidate,
        "prerequisites": normalized_prerequisites,
        "routes": expected_routes,
        "profiles": profiles,
        "cells": cells,
    }
    return normalized, path, _sha256_bytes(payload)


def _load_profiles(value: object) -> list[dict[str, Any]]:
    rows = _strict_sequence(value, "profiles", maximum=len(PROFILE_CONTRACTS))
    if len(rows) != len(PROFILE_CONTRACTS):
        raise MattePlatformQualificationError("profile catalog is incomplete")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _strict_mapping(
            value_row,
            {
                "id",
                "backend",
                "segmenter",
                "backend_kind",
                "quality_claim",
                "visual_candidate_id",
                "visual_algorithm_contract_sha256",
                "configured_policy_sha256",
                "effective_policy_sha256",
                "limits",
            },
            f"profiles[{index}]",
        )
        profile_id = _safe_id(row["id"], f"profiles[{index}].id")
        if profile_id in seen or profile_id not in PROFILE_CONTRACTS:
            raise MattePlatformQualificationError(
                "profile IDs must be exact and unique"
            )
        seen.add(profile_id)
        contract = PROFILE_CONTRACTS[profile_id]
        for key in ("backend", "segmenter", "backend_kind", "quality_claim"):
            if row[key] != contract[key]:
                raise MattePlatformQualificationError(
                    f"{profile_id} does not match its code-owned backend contract"
                )
        if contract["quality_claim"] is True:
            row["visual_candidate_id"] = _safe_id(
                row["visual_candidate_id"], f"{profile_id}.visual_candidate_id"
            )
            row["visual_algorithm_contract_sha256"] = _digest(
                row["visual_algorithm_contract_sha256"],
                f"{profile_id}.visual_algorithm_contract_sha256",
            )
        elif (
            row["visual_candidate_id"] is not None
            or row["visual_algorithm_contract_sha256"] is not None
        ):
            raise MattePlatformQualificationError(
                f"{profile_id} must not make an unqualified matte-quality claim"
            )
        row["configured_policy_sha256"] = _digest(
            row["configured_policy_sha256"], f"{profile_id}.configured_policy_sha256"
        )
        row["effective_policy_sha256"] = _digest(
            row["effective_policy_sha256"], f"{profile_id}.effective_policy_sha256"
        )
        limits = _strict_sequence(
            row["limits"], f"{profile_id}.limits", maximum=MAX_CELLS
        )
        normalized_limits = [
            _safe_id(item, f"{profile_id}.limits[{limit_index}]")
            for limit_index, item in enumerate(limits)
        ]
        if len(set(normalized_limits)) != len(normalized_limits):
            raise MattePlatformQualificationError(
                f"{profile_id}.limits contains duplicates"
            )
        row["limits"] = normalized_limits
        normalized.append(row)
    if seen != set(PROFILE_CONTRACTS):
        raise MattePlatformQualificationError("profile catalog is incomplete")
    quality_candidate_ids = [
        cast(str, row["visual_candidate_id"])
        for row in normalized
        if row["quality_claim"] is True
    ]
    if len(set(quality_candidate_ids)) != len(quality_candidate_ids):
        raise MattePlatformQualificationError(
            "quality backend tiers must bind distinct visual candidates"
        )
    return normalized


def _load_cells(
    value: object,
    root: Path,
    profiles: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    rows = _strict_sequence(value, "cells", maximum=MAX_CELLS)
    profile_ids = {cast(str, profile["id"]) for profile in profiles}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _strict_mapping(
            value_row,
            {
                "id",
                "route_id",
                "profile_id",
                "provider",
                "canvas",
                "target_fps",
                "dependency_profile",
                "state",
                "reason",
                "run",
                "capture_report",
            },
            f"cells[{index}]",
        )
        cell_id = _safe_id(row["id"], f"cells[{index}].id")
        if cell_id in seen:
            raise MattePlatformQualificationError("cell IDs must be unique")
        seen.add(cell_id)
        route_id = row["route_id"]
        profile_id = row["profile_id"]
        provider = row["provider"]
        dependency = row["dependency_profile"]
        if route_id not in ROUTE_CONTRACTS:
            raise MattePlatformQualificationError(f"{cell_id} route is invalid")
        if profile_id not in profile_ids:
            raise MattePlatformQualificationError(f"{cell_id} profile is invalid")
        if provider not in PROVIDERS:
            raise MattePlatformQualificationError(f"{cell_id} provider is invalid")
        if dependency not in DEPENDENCY_PROFILES:
            raise MattePlatformQualificationError(
                f"{cell_id} dependency profile is invalid"
            )
        canvas = _strict_mapping(
            row["canvas"], {"width", "height"}, f"{cell_id}.canvas"
        )
        width = _exact_int(
            canvas["width"], f"{cell_id}.canvas.width", minimum=1, maximum=4096
        )
        height = _exact_int(
            canvas["height"], f"{cell_id}.canvas.height", minimum=1, maximum=2160
        )
        if (width, height) not in CANVASES:
            raise MattePlatformQualificationError(f"{cell_id} canvas is unsupported")
        if row["target_fps"] != TARGET_FPS:
            raise MattePlatformQualificationError(
                f"{cell_id} must use the v1 balanced 30 FPS target"
            )
        contract = ROUTE_CONTRACTS[cast(str, route_id)]
        if provider == "directml" and contract["platform"] != "windows":
            raise MattePlatformQualificationError("DirectML is Windows-only")
        if provider == "cuda" and contract["platform"] == "darwin":
            raise MattePlatformQualificationError("CUDA is not a macOS provider")
        if profile_id != "rvm_matting" and provider != "cpu":
            raise MattePlatformQualificationError(
                "non-RVM backend tiers are qualified on their CPU provider"
            )
        if dependency == "without_mediapipe" and profile_id == "mediapipe_segmentation":
            raise MattePlatformQualificationError(
                "MediaPipe cannot be selected in the without-MediaPipe profile"
            )
        if dependency == "without_gpu_provider" and provider != "cpu":
            raise MattePlatformQualificationError(
                "a missing-GPU-provider profile must execute on CPU"
            )
        if route_id == "windows-native" and (width, height) not in (
            (1280, 720),
            (1920, 1080),
        ):
            raise MattePlatformQualificationError(
                "Windows native evidence must use a supported native mode"
            )
        if row["state"] not in ("recorded", "unavailable"):
            raise MattePlatformQualificationError(f"{cell_id}.state is invalid")
        if row["state"] == "recorded":
            if row["reason"] is not None:
                raise MattePlatformQualificationError(
                    f"{cell_id} recorded evidence cannot have an unavailable reason"
                )
            run = _descriptor(row["run"], root, f"{cell_id}.run")
            capture = _file_descriptor(
                row["capture_report"], root, f"{cell_id}.capture_report"
            )
        else:
            row["reason"] = _safe_id(row["reason"], f"{cell_id}.reason")
            if row["run"] is not None or row["capture_report"] is not None:
                raise MattePlatformQualificationError(
                    f"{cell_id} unavailable evidence must not name artifacts"
                )
            run = None
            capture = None
        normalized.append(
            {
                **row,
                "canvas": {"width": width, "height": height},
                "run": run,
                "capture_report": capture,
            }
        )
    return normalized


def _validate_matrix_coverage(
    profiles: Sequence[Mapping[str, object]],
    cells: Sequence[Mapping[str, object]],
) -> None:
    cell_ids = {cast(str, cell["id"]) for cell in cells}
    declared: list[str] = [
        cast(str, item)
        for profile in profiles
        for item in cast(Sequence[object], profile["limits"])
    ]
    if len(declared) != len(set(declared)) or set(declared) != cell_ids:
        raise MattePlatformQualificationError(
            "profile limits must reference every cell exactly once"
        )
    profile_by_id = {cast(str, row["id"]): row for row in profiles}
    for cell in cells:
        limits = cast(
            Sequence[object], profile_by_id[cast(str, cell["profile_id"])]["limits"]
        )
        if cell["id"] not in limits:
            raise MattePlatformQualificationError(
                "cell is assigned to the wrong profile"
            )
    if {cell["route_id"] for cell in cells} != set(ROUTE_CONTRACTS):
        raise MattePlatformQualificationError(
            "every reviewed route must remain visible"
        )
    if {cell["profile_id"] for cell in cells} != set(PROFILE_CONTRACTS):
        raise MattePlatformQualificationError("every backend tier must remain visible")
    if {cell["provider"] for cell in cells} != set(PROVIDERS):
        raise MattePlatformQualificationError(
            "CPU, CUDA, and DirectML must remain visible"
        )
    if {cell["dependency_profile"] for cell in cells} != set(DEPENDENCY_PROFILES):
        raise MattePlatformQualificationError(
            "standard and missing-dependency profiles must remain visible"
        )
    for profile_id in PROFILE_CONTRACTS:
        if not any(
            cell["profile_id"] == profile_id
            and cell["canvas"] == {"width": 1280, "height": 720}
            and cell["target_fps"] == TARGET_FPS
            for cell in cells
        ):
            raise MattePlatformQualificationError(
                f"{profile_id} lacks its required 1280x720@30 limit"
            )
    for route_id in ROUTE_CONTRACTS:
        if not any(
            cell["route_id"] == route_id and cell["dependency_profile"] == "standard"
            for cell in cells
        ):
            raise MattePlatformQualificationError(
                f"{route_id} lacks a standard dependency cell"
            )
    run_descriptors = [
        cast(Mapping[str, object], cell["run"])
        for cell in cells
        if cell["state"] == "recorded"
    ]
    if len({item["path_token"] for item in run_descriptors}) != len(
        run_descriptors
    ) or len({item["sha256"] for item in run_descriptors}) != len(run_descriptors):
        raise MattePlatformQualificationError(
            "recorded cells must use distinct run artifacts"
        )


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise MattePlatformQualificationError("a percentile requires samples")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    checked = [
        _finite(value, "timing sample", minimum=0.0, maximum=1_000_000.0)
        for value in values
    ]
    return {
        "count": len(checked),
        "p50": round(_nearest_rank(checked, 0.50), 6),
        "p95": round(_nearest_rank(checked, 0.95), 6),
        "p99": round(_nearest_rank(checked, 0.99), 6),
        "max": round(max(checked), 6),
    }


def _load_prerequisites(
    plan: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    descriptors = cast(Mapping[str, Mapping[str, object]], plan["prerequisites"])
    visual = _load_json_file(
        cast(Path, descriptors["visual"]["path"]),
        expected_sha256=cast(str, descriptors["visual"]["sha256"]),
        maximum=MAX_REPORT_BYTES,
    )
    if (
        visual.get("schema") != VISUAL_REPORT_SCHEMA
        or type(visual.get("version")) is not int
        or visual.get("version") != VISUAL_REPORT_VERSION
        or visual.get("evidence_sha256") != descriptors["visual"]["evidence_sha256"]
        or _report_digest(visual) != visual.get("evidence_sha256")
    ):
        raise MattePlatformQualificationError(
            "visual prerequisite is not an intact MATTE-5.2 report"
        )
    production = _strict_mapping(
        visual.get("production"),
        {
            "quality_preset_selected",
            "default_changed",
            "generated_evidence_can_qualify",
            "reactions_enabled",
        },
        "visual production disposition",
    )
    if (
        production["quality_preset_selected"] is not False
        or production["generated_evidence_can_qualify"] is not False
        or production["reactions_enabled"] is not False
        or production["default_changed"] is not False
    ):
        raise MattePlatformQualificationError(
            "visual prerequisite must be reaction-free and default-neutral"
        )
    manifest_rows = _strict_sequence(
        visual.get("algorithm_manifest"), "visual algorithm manifest", maximum=256
    )
    manifest: dict[str, Mapping[str, object]] = {}
    for row in manifest_rows:
        if not isinstance(row, Mapping):
            raise MattePlatformQualificationError(
                "visual algorithm manifest row is invalid"
            )
        candidate_id = _safe_id(row.get("id"), "visual candidate ID")
        if candidate_id in manifest:
            raise MattePlatformQualificationError(
                "visual algorithm manifest candidate IDs must be unique"
            )
        manifest[candidate_id] = row
    profiles = cast(Sequence[Mapping[str, object]], plan["profiles"])
    visual_authority: dict[str, Mapping[str, object]] = {}
    for profile in profiles:
        if profile["quality_claim"] is not True:
            continue
        candidate_id = cast(str, profile["visual_candidate_id"])
        candidate = manifest.get(candidate_id)
        if (
            candidate is None
            or candidate.get("contract_sha256")
            != profile["visual_algorithm_contract_sha256"]
        ):
            raise MattePlatformQualificationError(
                f"visual prerequisite does not bind {profile['id']}"
            )
        segmentation = candidate.get("segmentation")
        compositing = candidate.get("compositing")
        expected_backend = PROFILE_CONTRACTS[cast(str, profile["id"])]["backend"]
        if (
            not isinstance(segmentation, Mapping)
            or not isinstance(compositing, Mapping)
            or segmentation.get("backend") != expected_backend
        ):
            raise MattePlatformQualificationError(
                f"visual prerequisite binds {profile['id']} to the wrong backend"
            )
        configured_segmentation = dict(segmentation)
        configured_segmentation.pop("model_path_present", None)
        if profile["configured_policy_sha256"] != _canonical_digest(
            configured_segmentation
        ):
            raise MattePlatformQualificationError(
                f"{profile['id']} configured-policy digest is not visual-authoritative"
            )
        visual_authority[cast(str, profile["id"])] = candidate

    performance_raw = _load_json_file(
        cast(Path, descriptors["performance"]["path"]),
        expected_sha256=cast(str, descriptors["performance"]["sha256"]),
        maximum=MAX_REPORT_BYTES,
    )
    try:
        performance = validate_performance_report(performance_raw)
    except MattePerformanceError as exc:
        raise MattePlatformQualificationError(str(exc)) from exc
    if (
        performance.get("schema") != PERFORMANCE_REPORT_SCHEMA
        or type(performance.get("version")) is not int
        or performance.get("version") != PERFORMANCE_REPORT_VERSION
        or performance.get("evidence_sha256")
        != descriptors["performance"]["evidence_sha256"]
    ):
        raise MattePlatformQualificationError(
            "performance prerequisite is not an intact MATTE-3.4 report"
        )
    performance_scope = cast(Mapping[str, object], performance["scope"])
    performance_source = performance_scope.get("source")
    if (
        performance_scope.get("reactions_enabled") is not False
        or performance_scope.get("fixed_replay") is not True
        or not isinstance(performance_source, Mapping)
    ):
        raise MattePlatformQualificationError(
            "performance prerequisite must be a reaction-free fixed replay"
        )
    performance_source_sha256 = _digest(
        performance_scope.get("source_sha256"),
        "performance prerequisite source digest",
    )
    performance_lineage_sha256 = _digest(
        performance_source.get("measured_frame_lineage_sha256"),
        "performance prerequisite lineage digest",
    )
    performance_bundle_sha256 = performance_source.get("bundle_manifest_sha256")
    if performance_bundle_sha256 is not None:
        performance_bundle_sha256 = _digest(
            performance_bundle_sha256,
            "performance prerequisite bundle manifest digest",
        )
    if performance_source.get("execution_pacing") != "unpaced-tight-loop":
        raise MattePlatformQualificationError(
            "performance prerequisite fixed-replay acquisition is not exact"
        )
    performance_matrix = cast(Mapping[str, object], performance["matrix"])
    performance_rows = _strict_sequence(
        performance_matrix.get("rows"),
        "performance prerequisite matrix rows",
        maximum=64,
    )
    first_performance_row = cast(Mapping[str, object], performance_rows[0])
    performance_warmup = _exact_int(
        first_performance_row.get("warmup_frame_count"),
        "performance prerequisite warmup frames",
        minimum=0,
        maximum=MAX_SAMPLES,
    )
    performance_measured = _exact_int(
        first_performance_row.get("measured_frame_count"),
        "performance prerequisite measured frames",
        minimum=1,
        maximum=MAX_SAMPLES,
    )
    if any(
        not isinstance(row, Mapping)
        or row.get("warmup_frame_count") != performance_warmup
        or row.get("measured_frame_count") != performance_measured
        for row in performance_rows
    ):
        raise MattePlatformQualificationError(
            "performance prerequisite source spans disagree"
        )
    full_path = cast(Mapping[str, object], performance["full_path"])
    evidence = full_path.get("evidence")
    if isinstance(evidence, Mapping):
        effective = evidence.get("effective_policy")
        effective_sha256 = _sha256_bytes(_json_bytes(effective))
        rvm_profile = next(
            profile for profile in profiles if profile["id"] == "rvm_matting"
        )
        if effective_sha256 != rvm_profile["effective_policy_sha256"]:
            raise MattePlatformQualificationError(
                "MATTE-3.4 performance policy does not match the RVM tier"
            )
        if not isinstance(effective, Mapping):
            raise MattePlatformQualificationError(
                "MATTE-3.4 effective policy is malformed"
            )
        visual_rvm = visual_authority["rvm_matting"]
        rvm_segmentation = cast(Mapping[str, object], visual_rvm["segmentation"])
        rvm_compositing = cast(Mapping[str, object], visual_rvm["compositing"])
        configured_ratio = float(cast(float, rvm_segmentation["rvm_downsample"]))
        expected_ratio = 0.4 if configured_ratio == 0.0 else configured_ratio
        boundary_policy = cast(
            Mapping[str, object], rvm_segmentation["boundary_stabilization"]
        )
        wrap_policy = cast(
            Mapping[str, object], rvm_compositing["light_wrap_stabilization"]
        )
        color_policy = cast(Mapping[str, object], rvm_compositing["color_correction"])
        semantic_contract = {
            "resolved_rvm_downsample_ratio": expected_ratio,
            "raw_alpha_mode": "native_soft_alpha",
            "halo_mode": "mask_shift_only",
            "mask_shift_px": rvm_segmentation["mask_shift"],
            "boundary_stabilization_mode": boundary_policy["mode"],
            "use_model_foreground": rvm_compositing["use_model_foreground"],
            "light_wrap": rvm_compositing["light_wrap"],
            "light_wrap_stabilization_mode": wrap_policy["mode"],
            "blend_space": rvm_compositing["blend_space"],
            "color_correction_mode": color_policy["mode"],
        }
        if any(effective.get(key) != value for key, value in semantic_contract.items()):
            raise MattePlatformQualificationError(
                "MATTE-3.4 performance policy differs from the visual RVM contract"
            )

    rvm = _load_json_file(
        cast(Path, descriptors["rvm"]["path"]),
        expected_sha256=cast(str, descriptors["rvm"]["sha256"]),
        maximum=MAX_REPORT_BYTES,
    )
    if (
        rvm.get("schema") != RVM_REPORT_SCHEMA
        or type(rvm.get("version")) is not int
        or rvm.get("version") != RVM_REPORT_VERSION
        or rvm.get("evidence_sha256") != descriptors["rvm"]["evidence_sha256"]
    ):
        raise MattePlatformQualificationError(
            "RVM prerequisite is not an intact MATTE-2.5 report"
        )
    deterministic_keys = (
        "source",
        "provenance",
        "privacy",
        "qualification_contract",
        "coverage",
        "rows",
        "candidate_decisions",
        "profiles",
        "production",
    )
    if any(key not in rvm for key in deterministic_keys) or _sha256_bytes(
        _json_bytes({key: rvm[key] for key in deterministic_keys})
    ) != rvm.get("evidence_sha256"):
        raise MattePlatformQualificationError("RVM prerequisite digest is invalid")
    rvm_profiles = _strict_mapping(
        rvm["profiles"],
        {"status", "definitions", "reasons", "stable_cross_device_meaning_required"},
        "RVM profile disposition",
    )
    rvm_production = _strict_mapping(
        rvm["production"],
        {
            "default_changed",
            "high_detail_global_default_selected",
            "generated_proxy_can_select_profile",
        },
        "RVM production disposition",
    )
    if (
        rvm_production
        != {
            "default_changed": False,
            "high_detail_global_default_selected": False,
            "generated_proxy_can_select_profile": False,
        }
        or rvm_profiles["stable_cross_device_meaning_required"] is not True
    ):
        raise MattePlatformQualificationError(
            "RVM prerequisite changes defaults or weakens cross-device meaning"
        )
    rvm_profile = next(
        profile for profile in profiles if profile["id"] == "rvm_matting"
    )
    definitions = rvm_profiles["definitions"]
    selected_rvm_definition: Mapping[str, object] | None = None
    if rvm_profiles["status"] == "qualified":
        if not isinstance(definitions, Sequence) or isinstance(
            definitions, (str, bytes, bytearray)
        ):
            raise MattePlatformQualificationError(
                "qualified RVM prerequisite definitions are malformed"
            )
        matches = [
            item
            for item in definitions
            if isinstance(item, Mapping)
            and item.get("candidate_id") == rvm_profile["visual_candidate_id"]
        ]
        if len(matches) != 1:
            raise MattePlatformQualificationError(
                "qualified RVM prerequisite does not bind the visual RVM candidate"
            )
        selected_rvm_definition = cast(Mapping[str, object], matches[0])
        definition_segmentation = selected_rvm_definition.get("segmentation")
        definition_compositor = selected_rvm_definition.get("qualified_compositor")
        definition_model = selected_rvm_definition.get("model")
        definition_scope = selected_rvm_definition.get("qualification_scope")
        visual_rvm = visual_authority["rvm_matting"]
        visual_segmentation = dict(
            cast(Mapping[str, object], visual_rvm["segmentation"])
        )
        visual_segmentation.pop("model_path_present", None)
        if (
            not isinstance(definition_segmentation, Mapping)
            or not isinstance(definition_compositor, Mapping)
            or not isinstance(definition_model, Mapping)
            or not isinstance(definition_scope, Sequence)
            or isinstance(definition_scope, (str, bytes, bytearray))
            or not definition_scope
            or _canonical_digest(definition_segmentation)
            != selected_rvm_definition.get("segmentation_sha256")
            or _canonical_digest(definition_compositor)
            != selected_rvm_definition.get("qualified_compositor_sha256")
            or dict(definition_segmentation) != visual_segmentation
            or dict(definition_compositor)
            != dict(cast(Mapping[str, object], visual_rvm["compositing"]))
            or selected_rvm_definition.get("segmentation_sha256")
            != rvm_profile["configured_policy_sha256"]
        ):
            raise MattePlatformQualificationError(
                "qualified RVM definition differs from the selected visual policy"
            )

    visual_status = visual.get("status")
    if visual_status not in ("qualified", "pending", "failed"):
        raise MattePlatformQualificationError("visual prerequisite status is invalid")
    plan_kind = cast(Mapping[str, object], plan["provenance"])["kind"]
    if visual_status == "qualified":
        visual_provenance = cast(Mapping[str, object], visual.get("provenance"))
        visual_coverage = cast(Mapping[str, object], visual.get("coverage"))
        visual_quality = cast(Mapping[str, object], visual.get("quality"))
        visual_review = cast(Mapping[str, object], visual.get("review"))
        if not all(
            isinstance(item, Mapping)
            for item in (
                visual_provenance,
                visual_coverage,
                visual_quality,
                visual_review,
            )
        ) or (
            visual_provenance.get("kind") != plan_kind
            or plan_kind == "generated"
            or visual_coverage.get("complete") is not True
            or visual_coverage.get("boundaries_complete") is not True
            or visual_quality.get("absolute_passed") is not True
            or visual_quality.get("relative_nonregression_passed") is not True
            or visual_quality.get("material_improvement") is not True
            or visual_quality.get("optional_algorithms_passed") is not True
            or visual_review.get("all_cases_passed") is not True
            or visual_review.get("candidate_preferred_at_least_once") is not True
        ):
            raise MattePlatformQualificationError(
                "qualified visual prerequisite is internally inconsistent"
            )
        visual_cases = _strict_sequence(
            visual.get("cases"), "qualified visual cases", maximum=MAX_CELLS
        )
        case_by_candidate: dict[str, list[Mapping[str, object]]] = {}
        for raw_case in visual_cases:
            if not isinstance(raw_case, Mapping):
                raise MattePlatformQualificationError(
                    "qualified visual case is malformed"
                )
            candidate_id = raw_case.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise MattePlatformQualificationError(
                    "qualified visual case lacks a candidate"
                )
            case_by_candidate.setdefault(candidate_id, []).append(raw_case)
        expected_effective = {
            "rvm_matting": ("RVMSegmenter", "true_alpha_recurrent"),
            "mediapipe_segmentation": (
                "MediaPipeSegmenter",
                "confidence_mask_video",
            ),
        }
        for profile in profiles:
            if profile["quality_claim"] is not True:
                continue
            profile_id = cast(str, profile["id"])
            candidate_id = cast(str, profile["visual_candidate_id"])
            candidate = visual_authority[profile_id]
            candidate_cases = case_by_candidate.get(candidate_id, [])
            if not candidate_cases:
                raise MattePlatformQualificationError(
                    f"qualified visual prerequisite has no cases for {profile_id}"
                )
            valid_cases: list[Mapping[str, object]] = []
            for case in candidate_cases:
                contract = case.get("algorithm_contract")
                effective: object = None
                if isinstance(contract, Mapping):
                    effective = contract.get("expected_effective")
                policy = (
                    effective.get("matte_policy")
                    if isinstance(effective, Mapping)
                    else None
                )
                expected_backend_class, expected_backend_kind = expected_effective[
                    profile_id
                ]
                if (
                    case.get("outcome") != "passed"
                    or case.get("reactions_disabled") is not True
                    or not isinstance(contract, Mapping)
                    or contract.get("segmentation") != candidate.get("segmentation")
                    or contract.get("compositing") != candidate.get("compositing")
                    or not isinstance(effective, Mapping)
                    or effective.get("segmentation_backend") != expected_backend_class
                    or not isinstance(policy, Mapping)
                    or policy.get("selected_backend_kind") != expected_backend_kind
                ):
                    raise MattePlatformQualificationError(
                        f"qualified visual case contradicts {profile_id}"
                    )
                valid_cases.append(case)
            for cell in cast(Sequence[Mapping[str, object]], plan["cells"]):
                if cell["state"] != "recorded" or cell["profile_id"] != profile_id:
                    continue
                route = ROUTE_CONTRACTS[cast(str, cell["route_id"])]
                expected_platform = (
                    "macos" if route["platform"] == "darwin" else route["platform"]
                )
                boundary_id = (
                    "windows_native_loopback"
                    if cell["route_id"] == "windows-native"
                    else "pyvirtualcam_loopback"
                )
                expected_method = (
                    "windows-native-consumer-recording"
                    if boundary_id == "windows_native_loopback"
                    else "pyvirtualcam-consumer-recording"
                )
                authorized = False
                for case in valid_cases:
                    boundaries = case.get("boundaries")
                    boundary_rows = (
                        boundaries.get("boundaries")
                        if isinstance(boundaries, Mapping)
                        else None
                    )
                    boundary = (
                        boundary_rows.get(boundary_id)
                        if isinstance(boundary_rows, Mapping)
                        else None
                    )
                    comparison = (
                        boundary.get("comparison")
                        if isinstance(boundary, Mapping)
                        else None
                    )
                    coverage = case.get("coverage")
                    canvas = cast(Mapping[str, object], cell["canvas"])
                    canvas_id = f"{canvas['width']}x{canvas['height']}"
                    if (
                        isinstance(boundary, Mapping)
                        and isinstance(boundaries, Mapping)
                        and boundaries.get("authority") == "local-observed"
                        and boundary.get("status") == "passed"
                        and boundary.get("platform") == expected_platform
                        and boundary.get("capture_method") == expected_method
                        and isinstance(comparison, Mapping)
                        and comparison.get("width") == canvas["width"]
                        and comparison.get("height") == canvas["height"]
                        and isinstance(coverage, Mapping)
                        and isinstance(coverage.get("canvases"), list)
                        and canvas_id in coverage["canvases"]
                    ):
                        authorized = True
                        break
                if not authorized:
                    raise MattePlatformQualificationError(
                        f"visual prerequisite lacks {profile_id} sink evidence for "
                        f"{cell['route_id']}"
                    )
    performance_status = cast(Mapping[str, object], performance["decision"]).get(
        "outcome"
    )
    if performance_status not in ("qualified", "not_decidable", "rejected"):
        raise MattePlatformQualificationError(
            "performance prerequisite status is invalid"
        )
    rvm_status = rvm_profiles["status"]
    if rvm_status not in ("qualified", "not_decidable", "not_proposed"):
        raise MattePlatformQualificationError("RVM prerequisite status is invalid")
    if rvm_status == "qualified":
        rvm_provenance = cast(Mapping[str, object], rvm.get("provenance"))
        rvm_coverage = cast(Mapping[str, object], rvm.get("coverage"))
        rvm_rows = rvm.get("rows")
        if (
            not isinstance(rvm_provenance, Mapping)
            or not isinstance(rvm_coverage, Mapping)
            or not isinstance(rvm_rows, Sequence)
            or rvm_provenance.get("kind") != plan_kind
            or plan_kind == "generated"
            or rvm_coverage.get("complete") is not True
            or any(
                not isinstance(row, Mapping) or row.get("outcome") != "qualified"
                for row in rvm_rows
            )
        ):
            raise MattePlatformQualificationError(
                "qualified RVM prerequisite is internally inconsistent"
            )
    summaries: dict[str, object] = {
        "visual": {
            "file_sha256": descriptors["visual"]["sha256"],
            "evidence_sha256": visual["evidence_sha256"],
            "status": visual_status,
        },
        "performance": {
            "file_sha256": descriptors["performance"]["sha256"],
            "evidence_sha256": performance["evidence_sha256"],
            "status": performance_status,
            "fixed_replay_source_sha256": performance_source_sha256,
            "fixed_replay_bundle_manifest_sha256": performance_bundle_sha256,
            "fixed_replay_lineage_sha256": performance_lineage_sha256,
            "fixed_replay_warmup_frames": performance_warmup,
            "fixed_replay_measured_frames": performance_measured,
        },
        "rvm": {
            "file_sha256": descriptors["rvm"]["sha256"],
            "evidence_sha256": rvm["evidence_sha256"],
            "status": rvm_status,
            "selected_definition_sha256": (
                None
                if selected_rvm_definition is None
                else _canonical_digest(selected_rvm_definition)
            ),
        },
    }
    return (
        summaries,
        {
            "visual": cast(str, visual_status),
            "performance": cast(str, performance_status),
            "rvm": cast(str, rvm_status),
        },
        {
            "visual_candidates": visual_authority,
            "performance_source": {
                "source_sha256": performance_source_sha256,
                "bundle_manifest_sha256": performance_bundle_sha256,
                "measured_frame_lineage_sha256": performance_lineage_sha256,
                "warmup_frame_count": performance_warmup,
                "measured_frame_count": performance_measured,
            },
            "rvm_definition": selected_rvm_definition,
        },
    )


def _capture_report(
    descriptor: Mapping[str, object],
    cell: Mapping[str, object],
) -> tuple[dict[str, object], str, bool]:
    loaded = _load_json_file(
        cast(Path, descriptor["path"]),
        expected_sha256=cast(str, descriptor["sha256"]),
        maximum=MAX_REPORT_BYTES,
    )
    report = _strict_mapping(
        loaded,
        {
            "schema",
            "version",
            "privacy",
            "capture_only_contract",
            "condition",
            "requested",
            "negotiated",
            "camera_controls",
            "measurement",
            "timing",
            "pacing",
            "native_comparison",
            "full_runtime_comparison",
            "diagnosis",
            "qualification",
        },
        "capture prerequisite",
    )
    if (
        report.get("schema") != CAPTURE_REPORT_SCHEMA
        or type(report.get("version")) is not int
        or report.get("version") != CAPTURE_REPORT_VERSION
    ):
        raise MattePlatformQualificationError("capture prerequisite schema is invalid")
    privacy = _strict_mapping(
        report["privacy"],
        {
            "contains_pixels",
            "contains_frame_hashes",
            "contains_wall_clock_timestamps",
            "contains_device_path_or_index",
            "contains_only_opaque_identity_bindings",
            "contains_credentials",
            "timing_trace_uses_relative_monotonic_offsets",
        },
        "capture privacy",
    )
    if privacy != {
        "contains_pixels": False,
        "contains_frame_hashes": False,
        "contains_wall_clock_timestamps": False,
        "contains_device_path_or_index": False,
        "contains_only_opaque_identity_bindings": True,
        "contains_credentials": False,
        "timing_trace_uses_relative_monotonic_offsets": True,
    }:
        raise MattePlatformQualificationError("capture prerequisite is not private")
    contract = _strict_mapping(
        report.get("capture_only_contract"),
        {
            "production_capture_reader",
            "camera_acquisition",
            "canonical_normalization",
            "segmentation",
            "backdrop",
            "compositor",
            "preview",
            "api",
            "output_sink",
            "camera_control_policy",
            "camera_control_writes",
        },
        "capture-only contract",
    )
    if contract != {
        "production_capture_reader": True,
        "camera_acquisition": True,
        "canonical_normalization": True,
        "segmentation": False,
        "backdrop": False,
        "compositor": False,
        "preview": False,
        "api": False,
        "output_sink": False,
        "camera_control_policy": "preserve",
        "camera_control_writes": False,
    }:
        raise MattePlatformQualificationError("capture evidence is not capture-only")
    canvas = cast(Mapping[str, int], cell["canvas"])
    requested = cast(Mapping[str, object], report.get("requested"))
    negotiated = cast(Mapping[str, object], report.get("negotiated"))
    route = ROUTE_CONTRACTS[cast(str, cell["route_id"])]
    if not isinstance(requested, Mapping) or not isinstance(negotiated, Mapping):
        raise MattePlatformQualificationError("capture mode evidence is malformed")
    if (
        requested.get("width") != canvas["width"]
        or requested.get("height") != canvas["height"]
        or requested.get("canvas_width") != canvas["width"]
        or requested.get("canvas_height") != canvas["height"]
        or requested.get("fps") != cell["target_fps"]
        or str(negotiated.get("backend", "")).upper() != route["capture_backend"]
        or negotiated.get("delivered_width") != canvas["width"]
        or negotiated.get("delivered_height") != canvas["height"]
        or negotiated.get("normalized_width") != canvas["width"]
        or negotiated.get("normalized_height") != canvas["height"]
    ):
        raise MattePlatformQualificationError(
            "capture report does not match its route/canvas/FPS cell"
        )
    measurement = cast(Mapping[str, object], report.get("measurement"))
    timing = cast(Mapping[str, object], report.get("timing"))
    condition = cast(Mapping[str, object], report.get("condition"))
    if not all(isinstance(item, Mapping) for item in (measurement, timing, condition)):
        raise MattePlatformQualificationError("capture measurement is malformed")
    duration = _finite(
        measurement.get("measurement_seconds_actual"),
        "capture duration",
        minimum=0.001,
        maximum=86_400.0,
    )
    trace = _strict_sequence(
        timing.get("trace"), "capture timing trace", maximum=MAX_SAMPLES
    )
    if timing.get("sample_count") != len(trace):
        raise MattePlatformQualificationError("capture sample count is inconsistent")
    offsets: list[float] = []
    sequences: list[int] = []
    generations: list[int] = []
    geometry_generations: list[int] = []
    trace_timings: dict[str, list[float]] = {
        name: []
        for name in (
            "read_ms",
            "pre_normalization_ms",
            "normalization_ms",
            "publish_ms",
            "total_ms",
        )
    }
    reader_cpu: list[float | None] = []
    trace_keys = {
        "source_sequence_offset",
        "generation_offset",
        "geometry_generation_offset",
        "completion_offset_ms",
        "read_ms",
        "pre_normalization_ms",
        "normalization_ms",
        "publish_ms",
        "total_ms",
        "reader_cpu_ms",
    }
    for index, raw in enumerate(trace):
        item = _strict_mapping(raw, trace_keys, f"capture trace {index}")
        sequences.append(
            _exact_int(
                item.get("source_sequence_offset"),
                f"capture trace {index} sequence",
                minimum=0,
                maximum=2**63 - 1,
            )
        )
        generations.append(
            _exact_int(
                item.get("generation_offset"),
                f"capture trace {index} generation",
                minimum=0,
                maximum=2**31 - 1,
            )
        )
        geometry_generations.append(
            _exact_int(
                item.get("geometry_generation_offset"),
                f"capture trace {index} geometry generation",
                minimum=0,
                maximum=2**31 - 1,
            )
        )
        offsets.append(
            _finite(
                item.get("completion_offset_ms"),
                f"capture trace {index} offset",
                minimum=0.0,
                maximum=86_400_000.0,
            )
        )
        for timing_name in trace_timings:
            trace_timings[timing_name].append(
                _finite(
                    item.get(timing_name),
                    f"capture trace {index} {timing_name}",
                    maximum=60_000.0,
                )
            )
        cpu_value = item.get("reader_cpu_ms")
        reader_cpu.append(
            None
            if cpu_value is None
            else _finite(
                cpu_value,
                f"capture trace {index} reader_cpu_ms",
                maximum=60_000.0,
            )
        )
    if not offsets or sequences[0] != 0 or offsets[0] != 0.0:
        raise MattePlatformQualificationError(
            "capture trace must retain its zero sequence/time origin"
        )
    if any(current <= previous for previous, current in zip(sequences, sequences[1:])):
        raise MattePlatformQualificationError("capture sequence offsets must increase")
    if any(current <= previous for previous, current in zip(offsets, offsets[1:])):
        raise MattePlatformQualificationError(
            "capture completion offsets must increase"
        )
    source_success = sequences[-1] + 1
    observer_missing = sum(
        max(0, current - previous - 1)
        for previous, current in zip(sequences, sequences[1:])
    )
    if (
        timing.get("source_success_count") != source_success
        or timing.get("observer_missing_sample_count") != observer_missing
        or timing.get("generation_count") != len(set(generations))
        or timing.get("geometry_generation_count") != len(set(geometry_generations))
    ):
        raise MattePlatformQualificationError("capture timing identity was not derived")
    successful = _exact_int(
        measurement.get("successful_reads"),
        "capture successful reads",
        minimum=0,
        maximum=MAX_SAMPLES,
    )
    if successful != source_success:
        raise MattePlatformQualificationError("capture successful-read counts disagree")
    availability_fps = successful / duration
    window = cast(Mapping[str, object], timing.get("measurement_window"))
    if not isinstance(window, Mapping):
        raise MattePlatformQualificationError("capture measurement window is missing")
    window_duration = _finite(
        window.get("duration_seconds"), "capture window duration", minimum=0.001
    )
    leading_gap = _finite(window.get("leading_gap_ms"), "capture leading gap")
    trailing_gap = _finite(window.get("trailing_gap_ms"), "capture trailing gap")
    completion_span = _finite(
        window.get("completion_span_ms"), "capture completion span"
    )
    if (
        not math.isclose(window_duration, duration, rel_tol=0.0, abs_tol=1e-4)
        or not math.isclose(
            _finite(window.get("availability_fps"), "capture availability FPS"),
            availability_fps,
            rel_tol=0.0,
            abs_tol=1e-4,
        )
        or not math.isclose(completion_span, offsets[-1], rel_tol=0.0, abs_tol=1e-4)
        or not math.isclose(
            leading_gap + completion_span + trailing_gap,
            duration * 1000.0,
            rel_tol=0.0,
            abs_tol=1e-3,
        )
    ):
        raise MattePlatformQualificationError(
            "capture availability FPS was not derived"
        )
    first_per_generation: set[int] = set()
    steady_indexes: list[int] = []
    for index, generation in enumerate(generations):
        if generation in first_per_generation:
            steady_indexes.append(index)
        else:
            first_per_generation.add(generation)
    if not steady_indexes:
        steady_indexes = list(range(len(trace)))
    intervals: list[float] = []
    outages: list[float] = []
    for index in range(1, len(trace)):
        elapsed = offsets[index] - offsets[index - 1]
        if generations[index] == generations[index - 1]:
            intervals.append(elapsed / (sequences[index] - sequences[index - 1]))
        else:
            outages.append(elapsed)

    def require_summary(name: str, values: Sequence[float]) -> None:
        submitted = timing.get(name)
        if not isinstance(submitted, Mapping):
            raise MattePlatformQualificationError(f"capture {name} summary is missing")
        expected: dict[str, object]
        if values:
            expected = {
                "count": len(values),
                "p50": _nearest_rank(values, 0.50),
                "p95": _nearest_rank(values, 0.95),
                "p99": _nearest_rank(values, 0.99),
                "max": max(values),
            }
        else:
            expected = {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
        for key, expected_value in expected.items():
            actual = submitted.get(key)
            if expected_value is None:
                if actual is not None:
                    raise MattePlatformQualificationError(
                        f"capture {name}.{key} was not derived"
                    )
            elif key == "count":
                if actual != expected_value:
                    raise MattePlatformQualificationError(
                        f"capture {name}.{key} was not derived"
                    )
            elif not isinstance(actual, (int, float)) or not math.isclose(
                float(actual),
                float(cast(float, expected_value)),
                rel_tol=0.0,
                abs_tol=1e-4,
            ):
                raise MattePlatformQualificationError(
                    f"capture {name}.{key} was not derived"
                )

    require_summary("interval_ms", intervals)
    require_summary("cross_generation_outage_ms", outages)
    for timing_name, values in trace_timings.items():
        require_summary(timing_name, [values[index] for index in steady_indexes])
    require_summary(
        "reader_cpu_ms",
        [
            cast(float, reader_cpu[index])
            for index in steady_indexes
            if reader_cpu[index] is not None
        ],
    )
    total_span = offsets[-1] - offsets[0]
    wall_fps = (
        (sequences[-1] - sequences[0]) * 1000.0 / total_span
        if total_span > 0.0
        else None
    )
    active_span = sum(
        offsets[index] - offsets[index - 1]
        for index in range(1, len(offsets))
        if generations[index] == generations[index - 1]
    )
    active_successes = sum(
        sequences[index] - sequences[index - 1]
        for index in range(1, len(sequences))
        if generations[index] == generations[index - 1]
    )
    active_fps = active_successes * 1000.0 / active_span if active_span > 0.0 else None
    maximum_gap = max(intervals) if intervals else None
    for actual, expected, name in (
        (timing.get("wall_completion_fps"), wall_fps, "wall completion FPS"),
        (timing.get("active_capture_fps"), active_fps, "active capture FPS"),
        (timing.get("maximum_completion_gap_ms"), maximum_gap, "maximum gap"),
    ):
        if expected is None:
            valid = actual is None
        else:
            valid = isinstance(actual, (int, float)) and math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-4,
            )
        if not valid:
            raise MattePlatformQualificationError(f"capture {name} was not derived")
    latest_overwrites = _exact_int(
        measurement.get("latest_slot_overwrites"),
        "capture latest-slot overwrites",
        minimum=0,
        maximum=2**63 - 1,
    )
    for counter_name in (
        "read_failures",
        "restarts",
        "geometry_transitions",
        "warmup_read_failures",
        "warmup_restarts",
        "warmup_geometry_transitions",
    ):
        _exact_int(
            measurement.get(counter_name),
            f"capture {counter_name}",
            minimum=0,
            maximum=2**63 - 1,
        )
    device_digest = condition.get("device_identity_sha256")
    hardware_digest = condition.get("hardware_identity_sha256")
    identities_bound = False
    if device_digest is not None and hardware_digest is not None:
        _digest(device_digest, "capture device identity")
        _digest(hardware_digest, "capture hardware identity")
        identities_bound = True
    stable = all(
        measurement.get(key) in (0, None, False)
        for key in (
            "read_failures",
            "restarts",
            "geometry_transitions",
            "stalled_at_end",
            "capture_error",
            "close_error",
        )
    )
    threshold_fps = (
        float(cast(int, cell["target_fps"])) * MIN_CAPTURE_AND_OUTPUT_TARGET_RATIO
    )
    boundary_ms = 2.0 * 1000.0 / float(cast(int, cell["target_fps"]))
    rate_passed = (
        availability_fps >= threshold_fps
        and active_fps is not None
        and active_fps >= threshold_fps
        and wall_fps is not None
        and wall_fps >= threshold_fps
        and leading_gap <= boundary_ms
        and trailing_gap <= boundary_ms
        and maximum_gap is not None
        and maximum_gap <= boundary_ms
    )
    stable = stable and all(
        measurement.get(key) == 0
        for key in (
            "warmup_read_failures",
            "warmup_restarts",
            "warmup_geometry_transitions",
        )
    )
    hardware_attested = condition.get("hardware_verified") is True
    physical_source = condition.get("physical_source_eligible") is True
    raw_passed = bool(
        stable
        and rate_passed
        and hardware_attested
        and physical_source
        and identities_bound
        and observer_missing == 0
        and latest_overwrites == 0
        and len(set(generations)) == 1
        and len(set(geometry_generations)) == 1
        and duration >= 5.0
    )
    diagnosis = _strict_mapping(
        report["diagnosis"],
        {
            "code",
            "confidence",
            "actionable",
            "target_sustained",
            "target_threshold_fps",
            "capture_only_fps",
            "availability_fps",
            "measurement_window_complete",
            "measurement_window_sustained",
            "measurement_boundary_tolerance_ms",
            "minimum_successful_reads",
            "segmentation_executed",
            "segmentation_causal_for_capture_only_result",
            "runtime_comparison",
            "actions",
            "non_claims",
        },
        "capture diagnosis",
    )
    if (
        type(diagnosis["actionable"]) is not bool
        or type(diagnosis["target_sustained"]) is not bool
        or diagnosis["segmentation_executed"] is not False
        or diagnosis["segmentation_causal_for_capture_only_result"] is not False
        or not math.isclose(
            _finite(diagnosis["availability_fps"], "diagnosis availability FPS"),
            availability_fps,
            rel_tol=0.0,
            abs_tol=1e-4,
        )
    ):
        raise MattePlatformQualificationError("capture diagnosis is inconsistent")
    pacing = _strict_mapping(
        report["pacing"], {"capture", "processed_frames", "output"}, "capture pacing"
    )
    capture_pacing = _strict_mapping(
        pacing["capture"],
        {"measured", "active_source_fps", "wall_completion_fps", "target_fps"},
        "capture pacing.capture",
    )
    if (
        capture_pacing["measured"] is not True
        or capture_pacing["target_fps"] != cell["target_fps"]
        or not isinstance(capture_pacing["active_source_fps"], (int, float))
        or not math.isclose(
            float(capture_pacing["active_source_fps"]),
            cast(float, active_fps),
            rel_tol=0.0,
            abs_tol=1e-4,
        )
        or not isinstance(capture_pacing["wall_completion_fps"], (int, float))
        or not math.isclose(
            float(capture_pacing["wall_completion_fps"]),
            cast(float, wall_fps),
            rel_tol=0.0,
            abs_tol=1e-4,
        )
    ):
        raise MattePlatformQualificationError("capture pacing is inconsistent")
    processed_pacing = cast(Mapping[str, object], pacing["processed_frames"])
    output_pacing = cast(Mapping[str, object], pacing["output"])
    if (
        not isinstance(processed_pacing, Mapping)
        or processed_pacing.get("measured") is not False
        or not isinstance(output_pacing, Mapping)
        or output_pacing.get("measured_by_capture_only_harness") is not False
    ):
        raise MattePlatformQualificationError(
            "capture-only report contains processing or output pacing"
        )
    qualification = _strict_mapping(
        report["qualification"],
        {
            "backlog_acceptance_mode",
            "minimum_unique_fps",
            "exact_acceptance_mode_requested",
            "exact_mode_verified",
            "hardware_verified",
            "acceptance_satisfied",
            "outcome",
            "note",
        },
        "capture qualification",
    )
    official_outcome = qualification["outcome"]
    if official_outcome not in (
        "hardware-target-sustained",
        "hardware-actionable-limitation",
        "hardware-evidence-required",
    ) or (
        qualification["hardware_verified"] != hardware_attested
        or type(qualification["acceptance_satisfied"]) is not bool
        or (
            qualification["acceptance_satisfied"]
            != (
                official_outcome
                in ("hardware-target-sustained", "hardware-actionable-limitation")
            )
        )
    ):
        raise MattePlatformQualificationError(
            "capture qualification disposition is inconsistent"
        )
    if official_outcome == "hardware-target-sustained" and (
        diagnosis["code"] != "target-sustained"
        or diagnosis["target_sustained"] is not True
        or qualification["exact_acceptance_mode_requested"] is not True
        or qualification["exact_mode_verified"] is not True
    ):
        raise MattePlatformQualificationError(
            "capture target-sustained authority is inconsistent"
        )
    if (
        official_outcome == "hardware-actionable-limitation"
        and diagnosis["actionable"] is not True
    ):
        raise MattePlatformQualificationError(
            "capture actionable-limitation authority is inconsistent"
        )
    passed = bool(official_outcome == "hardware-target-sustained" and raw_passed)
    if passed:
        outcome = "qualified"
    elif official_outcome == "hardware-actionable-limitation":
        outcome = "pending"
    elif hardware_attested and not rate_passed:
        outcome = "failed"
    else:
        outcome = "pending"
    summary: dict[str, object] = {
        "file_sha256": descriptor["sha256"],
        "hardware_verified": hardware_attested,
        "device_identity_sha256": device_digest,
        "hardware_identity_sha256": hardware_digest,
        "duration_s": round(duration, 6),
        "unique_reads": successful,
        "availability_fps": round(availability_fps, 6),
        "interval_ms": timing.get("interval_ms"),
        "latest_slot_overwrites": latest_overwrites,
        "read_failures": measurement.get("read_failures"),
        "restarts": measurement.get("restarts"),
        "outcome": outcome,
        "matte31_outcome": official_outcome,
    }
    return summary, outcome, passed


def _load_run(
    descriptor: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    profile: Mapping[str, object],
    capture: Mapping[str, object],
    authorizations: Mapping[str, object],
) -> dict[str, Any]:
    run = _load_json_file(
        cast(Path, descriptor["path"]),
        expected_sha256=cast(str, descriptor["sha256"]),
        maximum=MAX_RUN_BYTES,
    )
    root = _strict_mapping(
        run,
        {
            "schema",
            "version",
            "provenance",
            "candidate",
            "model",
            "platform",
            "hardware",
            "runtime",
            "provider",
            "sink",
            "selection",
            "dependencies",
            "physical_authority",
            "capture_binding",
            "scope",
            "sources",
            "samples",
            "counters",
            "resource_samples",
            "lifecycle",
            "evidence_sha256",
        },
        "platform run evidence",
    )
    if (
        root["schema"] != RUN_SCHEMA
        or type(root["version"]) is not int
        or root["version"] != RUN_VERSION
    ):
        raise MattePlatformQualificationError("platform run evidence schema is invalid")
    if (
        root["evidence_sha256"] != descriptor["evidence_sha256"]
        or _report_digest(root) != root["evidence_sha256"]
    ):
        raise MattePlatformQualificationError("platform run evidence digest is invalid")

    provenance = _strict_mapping(
        root["provenance"],
        {"kind", "run_id", "build_sha256", "contains_pixels", "contains_paths"},
        "run provenance",
    )
    if provenance["kind"] != cast(Mapping[str, object], plan["provenance"])["kind"]:
        raise MattePlatformQualificationError("run and plan provenance disagree")
    provenance["run_id"] = _safe_id(provenance["run_id"], "run provenance.run_id")
    if (
        provenance["build_sha256"]
        != cast(Mapping[str, object], plan["candidate"])["build_sha256"]
    ):
        raise MattePlatformQualificationError("run and candidate build disagree")
    if (
        provenance["contains_pixels"] is not False
        or provenance["contains_paths"] is not False
    ):
        raise MattePlatformQualificationError(
            "run evidence must be path- and pixel-free"
        )

    candidate = _strict_mapping(
        root["candidate"],
        {
            "revision",
            "profile_id",
            "configured_policy_sha256",
            "effective_policy_sha256",
            "visual_candidate_id",
            "visual_algorithm_contract_sha256",
        },
        "run candidate",
    )
    if (
        candidate["revision"]
        != cast(Mapping[str, object], plan["candidate"])["revision"]
        or candidate["profile_id"] != cell["profile_id"]
        or candidate["configured_policy_sha256"] != profile["configured_policy_sha256"]
        or candidate["effective_policy_sha256"] != profile["effective_policy_sha256"]
        or candidate["visual_candidate_id"] != profile["visual_candidate_id"]
        or candidate["visual_algorithm_contract_sha256"]
        != profile["visual_algorithm_contract_sha256"]
    ):
        raise MattePlatformQualificationError(
            "run candidate does not match the declared visual/configured/effective tier"
        )

    route_id = cast(str, cell["route_id"])
    route = ROUTE_CONTRACTS[route_id]
    platform = _strict_mapping(
        root["platform"],
        {"route_id", "platform", "capture_backend", "sink_backend", "consumer"},
        "run platform",
    )
    if platform != {"route_id": route_id, **route}:
        raise MattePlatformQualificationError("run platform contract is invalid")

    hardware = _strict_mapping(
        root["hardware"],
        {"identity_sha256", "logical_cpu_count", "gpu_identity_sha256"},
        "run hardware",
    )
    hardware["identity_sha256"] = _digest(
        hardware["identity_sha256"], "run hardware.identity_sha256"
    )
    hardware["logical_cpu_count"] = _exact_int(
        hardware["logical_cpu_count"],
        "run hardware.logical_cpu_count",
        minimum=1,
        maximum=4096,
    )

    provider = _strict_mapping(
        root["provider"],
        {"name", "device", "device_id", "execution_verified"},
        "run provider",
    )
    if (
        provider["name"] != cell["provider"]
        or provider["execution_verified"] is not True
    ):
        raise MattePlatformQualificationError("run provider execution is not verified")
    provider["device"] = _safe_id(provider["device"], "run provider.device")
    if provider["name"] == "cpu":
        if (
            hardware["gpu_identity_sha256"] is not None
            or provider["device_id"] is not None
        ):
            raise MattePlatformQualificationError(
                "CPU evidence must not claim a GPU identity or device ordinal"
            )
    else:
        provider["device_id"] = _exact_int(
            provider["device_id"],
            "run provider.device_id",
            minimum=0,
            maximum=255,
        )
        hardware["gpu_identity_sha256"] = _digest(
            hardware["gpu_identity_sha256"], "run hardware.gpu_identity_sha256"
        )

    model_value = root["model"]
    if cell["profile_id"] == "rvm_matting":
        model = _strict_mapping(
            model_value,
            {"id", "sha256", "bytes"},
            "run RVM model",
        )
        model["id"] = _safe_text(model["id"], "run RVM model.id")
        model["sha256"] = _digest(model["sha256"], "run RVM model.sha256")
        model["bytes"] = _exact_int(
            model["bytes"],
            "run RVM model.bytes",
            minimum=1,
            maximum=2**63 - 1,
        )
    elif model_value is not None:
        raise MattePlatformQualificationError(
            "non-RVM platform evidence must mark the RVM model inapplicable"
        )
    else:
        model = None

    runtime = _strict_mapping(
        root["runtime"],
        {"python", "package_set_sha256", "provider_environment_sha256"},
        "run runtime",
    )
    runtime["python"] = _safe_text(runtime["python"], "run runtime.python")
    runtime["package_set_sha256"] = _digest(
        runtime["package_set_sha256"], "run runtime.package_set_sha256"
    )
    runtime["provider_environment_sha256"] = _digest(
        runtime["provider_environment_sha256"],
        "run runtime.provider_environment_sha256",
    )

    sink = _strict_mapping(
        root["sink"],
        {
            "backend",
            "consumer",
            "consumer_recording_verified",
            "no_unread_repeat_semantics",
        },
        "run sink",
    )
    if (
        sink["backend"] != route["sink_backend"]
        or sink["consumer"] != route["consumer"]
        or sink["no_unread_repeat_semantics"] is not True
        or type(sink["consumer_recording_verified"]) is not bool
    ):
        raise MattePlatformQualificationError("run sink contract is invalid")

    dependencies = _strict_mapping(
        root["dependencies"],
        {
            "profile",
            "mediapipe_installed",
            "gpu_provider_installed",
            "installed_distribution_sha256",
            "installed_distributions",
            "available_providers",
            "mediapipe_probe",
            "gpu_provider_probe",
        },
        "run dependencies",
    )
    if dependencies["profile"] != cell["dependency_profile"]:
        raise MattePlatformQualificationError("run dependency profile disagrees")
    for key in ("mediapipe_installed", "gpu_provider_installed"):
        if type(dependencies[key]) is not bool:
            raise MattePlatformQualificationError(
                f"run dependencies.{key} must be boolean"
            )
    dependencies["installed_distribution_sha256"] = _digest(
        dependencies["installed_distribution_sha256"],
        "run dependencies.installed_distribution_sha256",
    )
    installed_distributions = _strict_sequence(
        dependencies["installed_distributions"],
        "run installed distributions",
        maximum=512,
    )
    normalized_distributions = [
        _safe_text(item, f"run installed distribution {index}")
        for index, item in enumerate(installed_distributions)
    ]
    if normalized_distributions != sorted(
        normalized_distributions, key=str.casefold
    ) or len({item.casefold() for item in normalized_distributions}) != len(
        normalized_distributions
    ):
        raise MattePlatformQualificationError(
            "run installed distributions must be sorted and unique"
        )
    if dependencies["installed_distribution_sha256"] != _canonical_digest(
        normalized_distributions
    ):
        raise MattePlatformQualificationError(
            "run dependency digest does not bind its distribution inventory"
        )
    available_providers = _strict_sequence(
        dependencies["available_providers"],
        "run available providers",
        maximum=len(PROVIDER_RUNTIME_NAMES),
    )
    normalized_providers = [
        _safe_text(item, f"run available provider {index}")
        for index, item in enumerate(available_providers)
    ]
    allowed_provider_names = set(PROVIDER_RUNTIME_NAMES.values())
    if (
        normalized_providers != sorted(normalized_providers)
        or len(set(normalized_providers)) != len(normalized_providers)
        or not set(normalized_providers) <= allowed_provider_names
        or PROVIDER_RUNTIME_NAMES["cpu"] not in normalized_providers
    ):
        raise MattePlatformQualificationError(
            "run available-provider inventory is invalid"
        )
    mediapipe_present = any(
        item.casefold() == "mediapipe" for item in normalized_distributions
    )
    gpu_provider_present = any(
        PROVIDER_RUNTIME_NAMES[name] in normalized_providers
        for name in ("cuda", "directml")
    )
    if (
        dependencies["mediapipe_installed"] != mediapipe_present
        or dependencies["gpu_provider_installed"] != gpu_provider_present
        or dependencies["mediapipe_probe"]
        != ("available" if mediapipe_present else "missing")
        or dependencies["gpu_provider_probe"]
        != ("available" if gpu_provider_present else "missing")
    ):
        raise MattePlatformQualificationError(
            "run dependency claims were not derived from the inspected inventory"
        )
    dependencies["installed_distributions"] = normalized_distributions
    dependencies["available_providers"] = normalized_providers

    selection = _strict_mapping(
        root["selection"],
        {
            "requested_profile_id",
            "selected_profile_id",
            "fallback_reason",
            "fallback_visible",
            "slow_profile_suppressed",
        },
        "run selection",
    )
    if (
        selection["requested_profile_id"] not in PROFILE_CONTRACTS
        or selection["selected_profile_id"] != cell["profile_id"]
        or type(selection["fallback_visible"]) is not bool
        or type(selection["slow_profile_suppressed"]) is not bool
    ):
        raise MattePlatformQualificationError("run selection is invalid")
    dependency = cell["dependency_profile"]
    if dependency == "standard":
        if (
            selection["requested_profile_id"] != selection["selected_profile_id"]
            or selection["fallback_reason"] is not None
            or selection["fallback_visible"] is not False
            or selection["slow_profile_suppressed"] is not False
        ):
            raise MattePlatformQualificationError(
                "standard selection silently fell back"
            )
        if (
            cell["profile_id"] == "mediapipe_segmentation"
            and dependencies["mediapipe_installed"] is not True
        ):
            raise MattePlatformQualificationError(
                "MediaPipe execution lacks the MediaPipe package"
            )
    elif dependency == "without_mediapipe":
        if (
            dependencies["mediapipe_installed"] is not False
            or selection["requested_profile_id"] != "mediapipe_segmentation"
            or selection["fallback_reason"] != "mediapipe-unavailable"
            or selection["fallback_visible"] is not True
            or selection["slow_profile_suppressed"] is not True
        ):
            raise MattePlatformQualificationError(
                "missing-MediaPipe fallback is not visible and exact"
            )
    elif (
        dependencies["gpu_provider_installed"] is not False
        or selection["requested_profile_id"] != "rvm_matting"
        or selection["fallback_reason"] != "gpu-provider-unavailable"
        or selection["fallback_visible"] is not True
        or selection["slow_profile_suppressed"] is not True
    ):
        raise MattePlatformQualificationError(
            "missing-GPU-provider fallback is not visible and exact"
        )
    if provider["name"] != "cpu" and dependencies["gpu_provider_installed"] is not True:
        raise MattePlatformQualificationError(
            "GPU execution lacks its provider package"
        )
    if PROVIDER_RUNTIME_NAMES[cast(str, provider["name"])] not in normalized_providers:
        raise MattePlatformQualificationError(
            "selected execution provider is absent from the inspected inventory"
        )

    rvm_definition = authorizations.get("rvm_definition")
    if cell["profile_id"] == "rvm_matting" and isinstance(rvm_definition, Mapping):
        definition_model = rvm_definition.get("model")
        scopes = rvm_definition.get("qualification_scope")
        if not isinstance(definition_model, Mapping) or model != {
            "id": definition_model.get("id"),
            "sha256": definition_model.get("sha256"),
            "bytes": definition_model.get("bytes"),
        }:
            raise MattePlatformQualificationError(
                "run RVM model differs from its qualified profile definition"
            )
        if not isinstance(scopes, Sequence) or isinstance(
            scopes, (str, bytes, bytearray)
        ):
            raise MattePlatformQualificationError("qualified RVM scope is malformed")
        canvas = cast(Mapping[str, object], cell["canvas"])
        route_platform = ROUTE_CONTRACTS[cast(str, cell["route_id"])]["platform"]
        scope_matches = False
        for scope_row in scopes:
            if not isinstance(scope_row, Mapping):
                continue
            scope_hardware = scope_row.get("hardware")
            scope_provider = scope_row.get("provider")
            scope_canvas = scope_row.get("canvas")
            cadences = scope_row.get("cadences")
            render_modes = scope_row.get("render_modes")
            if (
                isinstance(scope_hardware, Mapping)
                and isinstance(scope_provider, Mapping)
                and isinstance(scope_canvas, Mapping)
                and scope_hardware.get("identity_sha256") == hardware["identity_sha256"]
                and _platform_family(scope_hardware.get("platform")) == route_platform
                and scope_provider.get("requested") == provider["name"]
                and scope_provider.get("active") == provider["name"]
                and scope_provider.get("execution_proven") is True
                and scope_provider.get("fallback_observed") is False
                and (
                    scope_provider.get("device_id") in (None, 0)
                    if provider["name"] == "cpu"
                    else scope_provider.get("device_id") == provider["device_id"]
                )
                and scope_provider.get("environment_sha256")
                == runtime["provider_environment_sha256"]
                and scope_canvas.get("width") == canvas["width"]
                and scope_canvas.get("height") == canvas["height"]
                and isinstance(cadences, list)
                and "native30" in cadences
                and isinstance(render_modes, list)
                and "qualified_compositor" in render_modes
            ):
                scope_matches = True
                break
        if not scope_matches:
            raise MattePlatformQualificationError(
                "run is outside the selected RVM definition qualification scope"
            )

    physical = _strict_mapping(
        root["physical_authority"],
        {
            "capture_origin_attested",
            "consumer_origin_attested",
            "same_host_attested",
            "cryptographically_proven",
        },
        "run physical authority",
    )
    for key in (
        "capture_origin_attested",
        "consumer_origin_attested",
        "same_host_attested",
        "cryptographically_proven",
    ):
        if type(physical[key]) is not bool:
            raise MattePlatformQualificationError(
                f"run physical_authority.{key} invalid"
            )
    if physical["cryptographically_proven"] is not False:
        raise MattePlatformQualificationError(
            "physical origin cannot be represented as cryptographically proven"
        )
    plan_provenance = cast(Mapping[str, object], plan["provenance"])
    for run_key, plan_key in (
        ("capture_origin_attested", "physical_capture_origin_attested"),
        ("consumer_origin_attested", "physical_consumer_origin_attested"),
        ("same_host_attested", "same_host_attested"),
    ):
        if physical[run_key] != plan_provenance[plan_key]:
            raise MattePlatformQualificationError(
                "run and plan physical authority disagree"
            )
    if sink["consumer_recording_verified"] != physical["consumer_origin_attested"]:
        raise MattePlatformQualificationError(
            "consumer recording and physical authority disagree"
        )

    capture_binding = _strict_mapping(
        root["capture_binding"],
        {
            "capture_file_sha256",
            "device_identity_sha256",
            "hardware_identity_sha256",
            "same_host_attested",
            "cryptographic_same_host_proof",
        },
        "run capture binding",
    )
    if (
        capture_binding["capture_file_sha256"] != capture["file_sha256"]
        or capture_binding["device_identity_sha256"]
        != capture["device_identity_sha256"]
        or capture_binding["hardware_identity_sha256"]
        != capture["hardware_identity_sha256"]
        or capture_binding["same_host_attested"] != physical["same_host_attested"]
        or capture_binding["cryptographic_same_host_proof"] is not False
    ):
        raise MattePlatformQualificationError(
            "run is not honestly bound to its capture-only report"
        )
    if physical["same_host_attested"] is True and (
        capture_binding["hardware_identity_sha256"] is None
        or hardware["identity_sha256"] != capture_binding["hardware_identity_sha256"]
    ):
        raise MattePlatformQualificationError(
            "same-host evidence uses different capture and processing identities"
        )

    scope = _strict_mapping(
        root["scope"],
        {
            "width",
            "height",
            "target_fps",
            "warmup_frames_by_source",
            "measured_seconds_by_source",
            "reactions_enabled",
            "post_base_event_count",
        },
        "run scope",
    )
    canvas = cast(Mapping[str, int], cell["canvas"])
    if (
        scope["width"] != canvas["width"]
        or scope["height"] != canvas["height"]
        or scope["target_fps"] != cell["target_fps"]
        or scope["reactions_enabled"] is not False
        or type(scope["post_base_event_count"]) is not int
        or scope["post_base_event_count"] != 0
    ):
        raise MattePlatformQualificationError(
            "run scope disagrees or enables reactions"
        )
    source_names = ("physical_capture", "fixed_replay")
    warmups = _strict_mapping(
        scope["warmup_frames_by_source"], set(source_names), "run warmup frames"
    )
    durations = _strict_mapping(
        scope["measured_seconds_by_source"], set(source_names), "run durations"
    )
    for source_name in source_names:
        warmups[source_name] = _exact_int(
            warmups[source_name],
            f"run {source_name} warmup frames",
            minimum=0,
            maximum=MAX_SAMPLES,
        )
        durations[source_name] = _finite(
            durations[source_name],
            f"run {source_name} measured seconds",
            minimum=0.001,
            maximum=86_400.0,
        )
    scope["warmup_frames_by_source"] = warmups
    scope["measured_seconds_by_source"] = durations

    source_contracts = _strict_mapping(
        root["sources"], set(source_names), "run source contracts"
    )
    physical_source = _strict_mapping(
        source_contracts["physical_capture"],
        {"kind", "acquisition", "capture_report_sha256", "trace_sha256"},
        "physical-capture source contract",
    )
    if (
        physical_source["kind"] != "physical-live-capture"
        or physical_source["acquisition"] != "paced-production-reader"
        or physical_source["capture_report_sha256"] != capture["file_sha256"]
    ):
        raise MattePlatformQualificationError(
            "physical-capture samples are not bound to the capture authority"
        )
    physical_source["trace_sha256"] = _digest(
        physical_source["trace_sha256"], "physical-capture trace digest"
    )
    fixed_source = _strict_mapping(
        source_contracts["fixed_replay"],
        {
            "kind",
            "acquisition",
            "source_sha256",
            "bundle_manifest_sha256",
            "measured_frame_lineage_sha256",
            "warmup_frame_count",
            "measured_frame_count",
            "measured_lineage",
            "trace_sha256",
        },
        "fixed-replay source contract",
    )
    performance_source = cast(
        Mapping[str, object], authorizations["performance_source"]
    )
    if (
        fixed_source["kind"] != "immutable-fixed-replay"
        or fixed_source["acquisition"] != "target-paced-replay"
        or fixed_source["source_sha256"] != performance_source["source_sha256"]
        or fixed_source["bundle_manifest_sha256"]
        != performance_source["bundle_manifest_sha256"]
        or fixed_source["measured_frame_lineage_sha256"]
        != performance_source["measured_frame_lineage_sha256"]
        or fixed_source["warmup_frame_count"]
        != performance_source["warmup_frame_count"]
        or fixed_source["measured_frame_count"]
        != performance_source["measured_frame_count"]
    ):
        raise MattePlatformQualificationError(
            "fixed-replay samples do not use the MATTE-3.4 source and lineage"
        )
    fixed_source["trace_sha256"] = _digest(
        fixed_source["trace_sha256"], "fixed-replay trace digest"
    )
    raw_lineage = _strict_sequence(
        fixed_source["measured_lineage"],
        "fixed-replay measured lineage",
        maximum=MAX_SAMPLES,
    )
    lineage: list[dict[str, int]] = []
    for index, raw in enumerate(raw_lineage):
        item = _strict_mapping(
            raw,
            {
                "capture_sequence",
                "capture_timestamp_ns",
                "capture_generation",
                "geometry_generation",
            },
            f"fixed-replay lineage {index}",
        )
        lineage.append(
            {
                "capture_sequence": _exact_int(
                    item["capture_sequence"],
                    f"fixed-replay lineage {index} capture sequence",
                    minimum=0,
                    maximum=2**63 - 1,
                ),
                "capture_timestamp_ns": _exact_int(
                    item["capture_timestamp_ns"],
                    f"fixed-replay lineage {index} capture timestamp",
                    minimum=0,
                    maximum=2**63 - 1,
                ),
                "capture_generation": _exact_int(
                    item["capture_generation"],
                    f"fixed-replay lineage {index} capture generation",
                    minimum=1,
                    maximum=2**31 - 1,
                ),
                "geometry_generation": _exact_int(
                    item["geometry_generation"],
                    f"fixed-replay lineage {index} geometry generation",
                    minimum=1,
                    maximum=2**31 - 1,
                ),
            }
        )
    if len(lineage) != fixed_source["measured_frame_count"]:
        raise MattePlatformQualificationError(
            "fixed-replay lineage length differs from MATTE-3.4"
        )
    lineage_digest = _canonical_digest(
        {
            "schema": "custback.matte-performance-measured-lineage",
            "version": 1,
            "source_sha256": fixed_source["source_sha256"],
            "warmup_frame_count": fixed_source["warmup_frame_count"],
            "measured_frame_count": fixed_source["measured_frame_count"],
            "frames": lineage,
        }
    )
    if lineage_digest != fixed_source["measured_frame_lineage_sha256"]:
        raise MattePlatformQualificationError(
            "fixed-replay lineage does not reproduce MATTE-3.4 authority"
        )
    fixed_source["measured_lineage"] = lineage

    sample_sets = _strict_mapping(root["samples"], set(source_names), "run samples")
    counter_sets = _strict_mapping(root["counters"], set(source_names), "run counters")
    samples: dict[str, list[dict[str, Any]]] = {}
    derived: dict[str, dict[str, Any]] = {}
    counter_keys = {
        "measured_frames",
        "unique_capture_frames",
        "unique_segmentations",
        "unique_composites",
        "sink_submissions",
        "capture_gaps",
        "capture_drops",
        "capture_slot_overwrites",
        "processing_deadline_misses",
        "sink_recoveries",
        "no_unread_repeats",
    }
    for source_name in source_names:
        samples[source_name], derived[source_name] = _load_samples(
            sample_sets[source_name],
            source_name=source_name,
            provider=cast(str, provider["name"]),
            logical_cpu_count=cast(int, hardware["logical_cpu_count"]),
            measured_seconds=cast(float, durations[source_name]),
            target_fps=cast(int, scope["target_fps"]),
            rvm=cell["profile_id"] == "rvm_matting",
        )
        submitted = _strict_mapping(
            counter_sets[source_name], counter_keys, f"run {source_name} counters"
        )
        if submitted != derived[source_name]["counters"]:
            raise MattePlatformQualificationError(
                f"run {source_name} counters were not derived from samples"
            )
        source_contract = cast(Mapping[str, object], source_contracts[source_name])
        if source_contract["trace_sha256"] != _canonical_digest(samples[source_name]):
            raise MattePlatformQualificationError(
                f"run {source_name} source contract does not bind its trace"
            )
        if source_name == "fixed_replay":
            if len(samples[source_name]) != len(lineage) or any(
                row["source_capture_sequence"] != lineage[index]["capture_sequence"]
                or row["source_capture_timestamp_ns"]
                != lineage[index]["capture_timestamp_ns"]
                or row["source_capture_generation"]
                != lineage[index]["capture_generation"]
                or row["source_geometry_generation"]
                != lineage[index]["geometry_generation"]
                for index, row in enumerate(samples[source_name])
            ):
                raise MattePlatformQualificationError(
                    "fixed-replay trace identities differ from MATTE-3.4 lineage"
                )
    if physical_source["trace_sha256"] == fixed_source["trace_sha256"]:
        raise MattePlatformQualificationError(
            "physical-capture and fixed-replay measurements must be independent"
        )

    resource_samples, resource = _load_resource_samples(
        root["resource_samples"],
        provider=cast(str, provider["name"]),
        logical_cpu_count=cast(int, hardware["logical_cpu_count"]),
    )
    lifecycle = _load_lifecycle(root["lifecycle"], provider=cast(str, provider["name"]))
    return {
        "file_sha256": descriptor["sha256"],
        "evidence_sha256": root["evidence_sha256"],
        "provenance": provenance,
        "candidate": candidate,
        "model": model,
        "platform": platform,
        "hardware": hardware,
        "runtime": runtime,
        "provider": provider,
        "sink": sink,
        "selection": selection,
        "dependencies": dependencies,
        "physical_authority": physical,
        "scope": scope,
        "sources": {
            "physical_capture": physical_source,
            "fixed_replay": fixed_source,
        },
        "samples": samples,
        "derived": derived,
        "resource_samples": resource_samples,
        "resource": resource,
        "lifecycle": lifecycle,
    }


def _load_samples(
    value: object,
    *,
    source_name: str,
    provider: str,
    logical_cpu_count: int,
    measured_seconds: float,
    target_fps: int,
    rvm: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _strict_sequence(value, "run samples", maximum=MAX_SAMPLES)
    sample_keys = {
        "index",
        "capture_sequence",
        "source_capture_sequence",
        "source_capture_timestamp_ns",
        "source_capture_generation",
        "source_geometry_generation",
        "segmentation_sequence",
        "composite_sequence",
        "output_sequence",
        "capture_generation",
        "capture_completed_ms",
        "processing_started_ms",
        "sink_completed_ms",
        "complete_service_ms",
        "serialized_cycle_ms",
        "frame_processing_ms",
        "model_preprocess_ms",
        "model_inference_ms",
        "model_postprocess_ms",
        "segmentation_ms",
        "refinement_ms",
        "background_ms",
        "compositor_ms",
        "compositor_substages_ms",
        "sink_prepare_ms",
        "sink_submit_ms",
        "sink_copy_ms",
        "pacing_wait_ms",
        "schedule_lateness_ms",
        "queue_age_ms",
        "end_to_end_age_ms",
        "process_cpu_percent",
        "rss_bytes",
        "gpu_utilization_percent",
        "vram_bytes",
        "capture_drop_count",
        "capture_slot_overwrite_count",
        "deadline_miss",
        "sink_recovery_count",
        "no_unread_repeat",
    }
    normalized: list[dict[str, Any]] = []
    timing_names = (
        "complete_service_ms",
        "serialized_cycle_ms",
        "frame_processing_ms",
        "segmentation_ms",
        "refinement_ms",
        "background_ms",
        "compositor_ms",
        "sink_prepare_ms",
        "sink_submit_ms",
        "sink_copy_ms",
        "pacing_wait_ms",
        "schedule_lateness_ms",
        "queue_age_ms",
        "end_to_end_age_ms",
    )
    model_names = (
        "model_preprocess_ms",
        "model_inference_ms",
        "model_postprocess_ms",
    )
    for index, raw in enumerate(rows):
        row = _strict_mapping(raw, sample_keys, f"run sample {index}")
        if row["index"] != index:
            raise MattePlatformQualificationError(
                "run sample indexes must be contiguous"
            )
        for name in (
            "capture_sequence",
            "source_capture_sequence",
            "source_capture_timestamp_ns",
            "segmentation_sequence",
            "composite_sequence",
            "output_sequence",
            "capture_generation",
            "source_capture_generation",
            "source_geometry_generation",
        ):
            row[name] = _exact_int(
                row[name],
                f"run sample {index} {name}",
                minimum=(
                    1
                    if name
                    in (
                        "capture_generation",
                        "source_capture_generation",
                        "source_geometry_generation",
                    )
                    else 0
                ),
                maximum=2**63 - 1,
            )
        for name in (
            "capture_completed_ms",
            "processing_started_ms",
            "sink_completed_ms",
        ):
            row[name] = _finite(
                row[name], f"run sample {index} {name}", maximum=86_400_000.0
            )
        if not (
            row["capture_completed_ms"]
            <= row["processing_started_ms"]
            <= row["sink_completed_ms"]
        ):
            raise MattePlatformQualificationError(
                "run sample timestamps are misordered"
            )
        for name in timing_names:
            row[name] = _finite(
                row[name], f"run sample {index} {name}", maximum=60_000.0
            )
        for name in model_names:
            if rvm:
                row[name] = _finite(
                    row[name], f"run sample {index} {name}", maximum=60_000.0
                )
            elif row[name] is not None:
                raise MattePlatformQualificationError(
                    "non-RVM samples must mark RVM model stages inapplicable"
                )
        substages = _strict_mapping(
            row["compositor_substages_ms"],
            set(COMPOSITOR_SUBSTAGE_NAMES),
            f"run sample {index} compositor substages",
        )
        row["compositor_substages_ms"] = {
            name: _finite(
                substages[name],
                f"run sample {index} compositor {name}",
                maximum=60_000.0,
            )
            for name in COMPOSITOR_SUBSTAGE_NAMES
        }
        if sum(cast(dict[str, float], row["compositor_substages_ms"]).values()) > (
            row["compositor_ms"] + 0.01
        ):
            raise MattePlatformQualificationError(
                "compositor substages exceed their total"
            )
        model_total = sum(
            float(row[name]) for name in model_names if row[name] is not None
        )
        major_processing = sum(
            float(row[name])
            for name in (
                "segmentation_ms",
                "refinement_ms",
                "background_ms",
                "compositor_ms",
            )
        )
        sink_total = sum(
            float(row[name])
            for name in ("sink_prepare_ms", "sink_submit_ms", "sink_copy_ms")
        )
        if (
            model_total > float(row["segmentation_ms"]) + 0.01
            or major_processing > float(row["frame_processing_ms"]) + 0.01
            or float(row["frame_processing_ms"]) + sink_total
            > float(row["complete_service_ms"]) + 0.01
            or float(row["complete_service_ms"])
            > float(row["serialized_cycle_ms"]) + 0.01
            or float(row["complete_service_ms"]) + float(row["pacing_wait_ms"])
            > float(row["serialized_cycle_ms"]) + 0.01
        ):
            raise MattePlatformQualificationError(
                "run timing scopes are arithmetically inconsistent"
            )
        service_from_clock = row["sink_completed_ms"] - row["processing_started_ms"]
        age_from_clock = row["sink_completed_ms"] - row["capture_completed_ms"]
        queue_from_clock = row["processing_started_ms"] - row["capture_completed_ms"]
        for actual, expected, name in (
            (row["complete_service_ms"], service_from_clock, "complete service"),
            (row["end_to_end_age_ms"], age_from_clock, "end-to-end age"),
            (row["queue_age_ms"], queue_from_clock, "queue age"),
        ):
            if not math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=0.01
            ):
                raise MattePlatformQualificationError(
                    f"run sample {index} {name} was not derived from timestamps"
                )
        row["process_cpu_percent"] = _finite(
            row["process_cpu_percent"],
            f"run sample {index} process CPU",
            maximum=float(logical_cpu_count * 100),
        )
        row["rss_bytes"] = _exact_int(
            row["rss_bytes"],
            f"run sample {index} RSS",
            minimum=1,
            maximum=2**63 - 1,
        )
        if provider == "cpu":
            if (
                row["gpu_utilization_percent"] is not None
                or row["vram_bytes"] is not None
            ):
                raise MattePlatformQualificationError(
                    "CPU samples must mark GPU utilization and VRAM inapplicable"
                )
        else:
            row["gpu_utilization_percent"] = _finite(
                row["gpu_utilization_percent"],
                f"run sample {index} GPU utilization",
                maximum=100.0,
            )
            row["vram_bytes"] = _exact_int(
                row["vram_bytes"],
                f"run sample {index} VRAM",
                minimum=1,
                maximum=2**63 - 1,
            )
        for name in (
            "capture_drop_count",
            "capture_slot_overwrite_count",
            "sink_recovery_count",
        ):
            row[name] = _exact_int(
                row[name],
                f"run sample {index} {name}",
                minimum=0,
                maximum=2**31 - 1,
            )
        if (
            type(row["deadline_miss"]) is not bool
            or type(row["no_unread_repeat"]) is not bool
        ):
            raise MattePlatformQualificationError(
                "run sample counter flags must be boolean"
            )
        expected_deadline = row["complete_service_ms"] > (1000.0 / target_fps)
        if row["deadline_miss"] != expected_deadline:
            raise MattePlatformQualificationError(
                "processing deadline flags were not derived from service time"
            )
        normalized.append(row)

    if source_name == "physical_capture" and any(
        row["source_capture_sequence"] != row["capture_sequence"]
        or row["source_capture_generation"] != row["capture_generation"]
        for row in normalized
    ):
        raise MattePlatformQualificationError(
            "physical-capture local identities differ from their source identities"
        )

    if len(normalized) < MIN_MEASURED_FRAMES:
        # Short evidence is well-formed but cannot pass.  Keep it for a
        # transparent failed/pending decision instead of treating it as JSON
        # corruption.
        pass
    for previous, current in zip(normalized, normalized[1:]):
        if (
            current["capture_sequence"] < previous["capture_sequence"]
            or current["segmentation_sequence"] < previous["segmentation_sequence"]
            or current["composite_sequence"] < previous["composite_sequence"]
            or current["output_sequence"] != previous["output_sequence"] + 1
            or current["capture_completed_ms"] < previous["capture_completed_ms"]
            or current["processing_started_ms"] < previous["processing_started_ms"]
            or current["sink_completed_ms"] <= previous["sink_completed_ms"]
        ):
            raise MattePlatformQualificationError(
                "run sample identity/timestamps regress"
            )
        if cast(float, current["processing_started_ms"]) < cast(
            float, previous["sink_completed_ms"]
        ):
            raise MattePlatformQualificationError("serialized run samples overlap")
        derived_cycle = cast(float, current["sink_completed_ms"]) - cast(
            float, previous["sink_completed_ms"]
        )
        derived_wait = cast(float, current["processing_started_ms"]) - cast(
            float, previous["sink_completed_ms"]
        )
        if not math.isclose(
            cast(float, current["serialized_cycle_ms"]),
            derived_cycle,
            rel_tol=0.0,
            abs_tol=0.01,
        ) or not math.isclose(
            cast(float, current["pacing_wait_ms"]),
            derived_wait,
            rel_tol=0.0,
            abs_tol=0.01,
        ):
            raise MattePlatformQualificationError(
                "serialized cycle or pacing wait was not derived from timestamps"
            )
        capture_delta = current["capture_sequence"] - previous["capture_sequence"]
        if (capture_delta == 0) != (
            current["capture_completed_ms"] == previous["capture_completed_ms"]
        ):
            raise MattePlatformQualificationError(
                "capture identity and completion time disagree"
            )
        if source_name == "physical_capture" and (
            (capture_delta == 0)
            != (
                current["source_capture_timestamp_ns"]
                == previous["source_capture_timestamp_ns"]
            )
        ):
            raise MattePlatformQualificationError(
                "physical source identity and timestamp disagree"
            )
        if current["capture_slot_overwrite_count"] > max(0, capture_delta - 1):
            raise MattePlatformQualificationError(
                "capture-slot overwrites exceed the observed sequence gap"
            )
        repeated = current["composite_sequence"] == previous["composite_sequence"]
        if current["no_unread_repeat"] != repeated:
            raise MattePlatformQualificationError(
                "no-unread repeat flags do not match composite identity"
            )
        if (
            not repeated
            and current["composite_sequence"] != previous["composite_sequence"] + 1
        ):
            raise MattePlatformQualificationError(
                "composite sequence has an unexplained gap"
            )
        segmentation_delta = (
            current["segmentation_sequence"] - previous["segmentation_sequence"]
        )
        composite_delta = current["composite_sequence"] - previous["composite_sequence"]
        expected_work_delta = 1 if capture_delta > 0 else 0
        if (
            segmentation_delta != expected_work_delta
            or composite_delta != expected_work_delta
        ):
            raise MattePlatformQualificationError(
                "segmentation/composite work is not caused by a new capture"
            )
    if normalized:
        first = normalized[0]
        if (
            first["capture_sequence"] != 0
            or first["segmentation_sequence"] != 0
            or first["composite_sequence"] != 0
            or first["output_sequence"] != 0
            or first["capture_generation"] != 1
            or first["capture_completed_ms"] != 0.0
            or first["no_unread_repeat"] is not False
            or first["capture_drop_count"] != 0
            or first["capture_slot_overwrite_count"] != 0
            or first["sink_recovery_count"] != 0
            or first["sink_completed_ms"] > (2.0 * 1000.0 / target_fps)
            or not math.isclose(
                cast(float, first["serialized_cycle_ms"]),
                cast(float, first["sink_completed_ms"]),
                rel_tol=0.0,
                abs_tol=0.01,
            )
            or not math.isclose(
                cast(float, first["pacing_wait_ms"]),
                cast(float, first["processing_started_ms"]),
                rel_tol=0.0,
                abs_tol=0.01,
            )
            or not math.isclose(
                cast(float, first["serialized_cycle_ms"]),
                1000.0 / target_fps,
                rel_tol=0.0,
                abs_tol=0.01,
            )
            or not math.isclose(
                cast(float, first["pacing_wait_ms"]),
                max(
                    0.0,
                    1000.0 / target_fps - cast(float, first["complete_service_ms"]),
                ),
                rel_tol=0.0,
                abs_tol=0.01,
            )
        ):
            raise MattePlatformQualificationError(
                "run trace must retain a zero origin and zero initial event increments"
            )
        if len({row["capture_generation"] for row in normalized}) != 1:
            raise MattePlatformQualificationError(
                "steady measurement cannot hide a capture-generation transition"
            )
        if (
            source_name == "physical_capture"
            and len({row["source_geometry_generation"] for row in normalized}) != 1
        ):
            raise MattePlatformQualificationError(
                "steady physical measurement cannot hide a geometry transition"
            )
        schedule_origin = float(first["sink_completed_ms"])
        frame_ms = 1000.0 / target_fps
        for index, row in enumerate(normalized):
            scheduled_ms = schedule_origin + index * frame_ms
            derived_lateness = max(0.0, float(row["sink_completed_ms"]) - scheduled_ms)
            if not math.isclose(
                float(row["schedule_lateness_ms"]),
                derived_lateness,
                rel_tol=0.0,
                abs_tol=0.01,
            ):
                raise MattePlatformQualificationError(
                    "schedule lateness was not derived from the target cadence"
                )
        last_sink = float(normalized[-1]["sink_completed_ms"])
        window_end = measured_seconds * 1000.0
        if not window_end - frame_ms * 2.0 <= last_sink <= window_end + frame_ms * 2.0:
            raise MattePlatformQualificationError(
                "run samples do not span the declared measurement window"
            )

    unique_capture = len({row["capture_sequence"] for row in normalized})
    unique_segmentations = len({row["segmentation_sequence"] for row in normalized})
    unique_composites = len({row["composite_sequence"] for row in normalized})
    capture_gaps = sum(
        max(0, current["capture_sequence"] - previous["capture_sequence"] - 1)
        for previous, current in zip(normalized, normalized[1:])
    )
    counters = {
        "measured_frames": len(normalized),
        "unique_capture_frames": unique_capture,
        "unique_segmentations": unique_segmentations,
        "unique_composites": unique_composites,
        "sink_submissions": len(normalized),
        "capture_gaps": capture_gaps,
        "capture_drops": sum(row["capture_drop_count"] for row in normalized),
        "capture_slot_overwrites": sum(
            row["capture_slot_overwrite_count"] for row in normalized
        ),
        "processing_deadline_misses": sum(
            1 for row in normalized if row["deadline_miss"]
        ),
        "sink_recoveries": sum(row["sink_recovery_count"] for row in normalized),
        "no_unread_repeats": sum(1 for row in normalized if row["no_unread_repeat"]),
    }
    intervals = [
        current["sink_completed_ms"] - previous["sink_completed_ms"]
        for previous, current in zip(normalized, normalized[1:])
    ]
    jitter = [abs(value - 1000.0 / target_fps) for value in intervals]
    unique_capture_events = [
        row
        for index, row in enumerate(normalized)
        if index == 0
        or row["capture_sequence"] != normalized[index - 1]["capture_sequence"]
    ]
    unique_composite_events = [
        row
        for index, row in enumerate(normalized)
        if index == 0
        or row["composite_sequence"] != normalized[index - 1]["composite_sequence"]
    ]
    unique_capture_intervals = [
        cast(float, current["sink_completed_ms"])
        - cast(float, previous["sink_completed_ms"])
        for previous, current in zip(unique_capture_events, unique_capture_events[1:])
    ]
    unique_composite_intervals = [
        cast(float, current["sink_completed_ms"])
        - cast(float, previous["sink_completed_ms"])
        for previous, current in zip(
            unique_composite_events, unique_composite_events[1:]
        )
    ]
    if unique_capture_events:
        unique_capture_intervals.extend(
            (
                cast(float, unique_capture_events[0]["sink_completed_ms"]),
                max(
                    0.0,
                    measured_seconds * 1000.0
                    - cast(float, unique_capture_events[-1]["sink_completed_ms"]),
                ),
            )
        )
    if unique_composite_events:
        unique_composite_intervals.extend(
            (
                cast(float, unique_composite_events[0]["sink_completed_ms"]),
                max(
                    0.0,
                    measured_seconds * 1000.0
                    - cast(float, unique_composite_events[-1]["sink_completed_ms"]),
                ),
            )
        )
    timing_fields = (
        *model_names,
        *timing_names,
        "process_cpu_percent",
        "gpu_utilization_percent",
    )
    timings: dict[str, object] = {}
    for name in timing_fields:
        values = [
            float(row[name])
            for row in normalized
            if isinstance(row[name], (int, float)) and not isinstance(row[name], bool)
        ]
        timings[name] = _summary(values) if values else None
    for name in COMPOSITOR_SUBSTAGE_NAMES:
        timings[f"compositor.{name}"] = _summary(
            [float(row["compositor_substages_ms"][name]) for row in normalized]
        )
    timings["output_interval_ms"] = _summary(intervals) if intervals else None
    timings["inter_frame_jitter_ms"] = _summary(jitter) if jitter else None
    timings["unique_capture_interval_ms"] = (
        _summary(unique_capture_intervals) if unique_capture_intervals else None
    )
    timings["unique_composite_interval_ms"] = (
        _summary(unique_composite_intervals) if unique_composite_intervals else None
    )
    first_third = max(1, len(normalized) // 3)
    last_third = max(1, len(normalized) // 3)
    e2e_head = [float(row["end_to_end_age_ms"]) for row in normalized[:first_third]]
    e2e_tail = [float(row["end_to_end_age_ms"]) for row in normalized[-last_third:]]
    e2e_drift = (
        _nearest_rank(e2e_tail, 0.95) - _nearest_rank(e2e_head, 0.95)
        if normalized
        else 0.0
    )
    rates = {
        "capture_fps": round(unique_capture / measured_seconds, 6),
        "segmentation_fps": round(unique_segmentations / measured_seconds, 6),
        "unique_composite_fps": round(unique_composites / measured_seconds, 6),
        "output_fps": round(len(normalized) / measured_seconds, 6),
    }
    return normalized, {
        "counters": counters,
        "rates": rates,
        "timings": timings,
        "e2e_age_p95_drift_ms": round(e2e_drift, 6),
    }


def _load_resource_samples(
    value: object,
    *,
    provider: str,
    logical_cpu_count: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = _strict_sequence(value, "resource samples", maximum=MAX_SAMPLES)
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(rows):
        row = _strict_mapping(
            raw,
            {
                "offset_s",
                "process_cpu_percent",
                "rss_bytes",
                "gpu_percent",
                "vram_bytes",
            },
            f"resource sample {index}",
        )
        row["offset_s"] = _finite(
            row["offset_s"], f"resource sample {index} offset", maximum=604_800.0
        )
        row["process_cpu_percent"] = _finite(
            row["process_cpu_percent"],
            f"resource sample {index} CPU",
            maximum=float(logical_cpu_count * 100),
        )
        row["rss_bytes"] = _exact_int(
            row["rss_bytes"],
            f"resource sample {index} RSS",
            minimum=1,
            maximum=2**63 - 1,
        )
        if provider == "cpu":
            if row["gpu_percent"] is not None or row["vram_bytes"] is not None:
                raise MattePlatformQualificationError(
                    "CPU resource samples must mark GPU fields inapplicable"
                )
        else:
            row["gpu_percent"] = _finite(
                row["gpu_percent"],
                f"resource sample {index} GPU",
                maximum=100.0,
            )
            row["vram_bytes"] = _exact_int(
                row["vram_bytes"],
                f"resource sample {index} VRAM",
                minimum=1,
                maximum=2**63 - 1,
            )
        normalized.append(row)
    if any(
        cast(float, current["offset_s"]) <= cast(float, previous["offset_s"])
        for previous, current in zip(normalized, normalized[1:])
    ):
        raise MattePlatformQualificationError("resource sample offsets must increase")
    if normalized[0]["offset_s"] != 0.0:
        raise MattePlatformQualificationError(
            "resource samples must retain a zero origin"
        )
    if len(normalized) < 2:
        raise MattePlatformQualificationError(
            "resource evidence requires at least two observations"
        )
    span = cast(float, normalized[-1]["offset_s"]) - cast(
        float, normalized[0]["offset_s"]
    )
    head_limit = span * 0.25
    tail_limit = span * 0.75
    head_rows = [
        row for row in normalized if cast(float, row["offset_s"]) <= head_limit
    ]
    tail_rows = [
        row for row in normalized if cast(float, row["offset_s"]) >= tail_limit
    ]
    if len(head_rows) < 2 or len(tail_rows) < 2:
        raise MattePlatformQualificationError(
            "resource evidence lacks time-distributed head/tail observations"
        )
    rss_head = [cast(int, row["rss_bytes"]) for row in head_rows]
    rss_tail = [cast(int, row["rss_bytes"]) for row in tail_rows]
    rss_drift = _nearest_rank(rss_tail, 0.95) - _nearest_rank(rss_head, 0.50)
    if provider == "cpu":
        vram_drift: float | None = None
    else:
        vram_head = [cast(int, row["vram_bytes"]) for row in head_rows]
        vram_tail = [cast(int, row["vram_bytes"]) for row in tail_rows]
        vram_drift = _nearest_rank(vram_tail, 0.95) - _nearest_rank(vram_head, 0.50)
    sample_gaps = [
        cast(float, current["offset_s"]) - cast(float, previous["offset_s"])
        for previous, current in zip(normalized, normalized[1:])
    ]
    maximum_sample_gap = max(sample_gaps)
    nominal_sample_gap = _nearest_rank(sample_gaps, 0.50)
    if any(
        gap < nominal_sample_gap * 0.50 or gap > nominal_sample_gap * 1.50
        for gap in sample_gaps
    ):
        raise MattePlatformQualificationError(
            "resource observations do not use a bounded uniform cadence"
        )
    return normalized, {
        "sample_count": len(normalized),
        "span_s": round(span, 6),
        "maximum_sample_gap_s": round(maximum_sample_gap, 6),
        "nominal_sample_gap_s": round(nominal_sample_gap, 6),
        "process_cpu_percent": _summary(
            [cast(float, row["process_cpu_percent"]) for row in normalized]
        ),
        "rss_bytes": {
            "first_quarter_p50": round(_nearest_rank(rss_head, 0.50), 3),
            "last_quarter_p95": round(_nearest_rank(rss_tail, 0.95), 3),
            "signed_drift_bytes": round(rss_drift, 3),
            "span_bytes": max(cast(int, row["rss_bytes"]) for row in normalized)
            - min(cast(int, row["rss_bytes"]) for row in normalized),
            "max": max(cast(int, row["rss_bytes"]) for row in normalized),
        },
        "gpu_percent": (
            None
            if provider == "cpu"
            else _summary([cast(float, row["gpu_percent"]) for row in normalized])
        ),
        "vram_bytes": (
            None
            if provider == "cpu"
            else {
                "signed_drift_bytes": round(cast(float, vram_drift), 3),
                "span_bytes": max(cast(int, row["vram_bytes"]) for row in normalized)
                - min(cast(int, row["vram_bytes"]) for row in normalized),
                "max": max(cast(int, row["vram_bytes"]) for row in normalized),
            }
        ),
    }


def _load_lifecycle(value: object, *, provider: str) -> dict[str, Any]:
    lifecycle = _strict_mapping(
        value,
        {
            "soak_duration_s",
            "events",
            "counter_snapshots",
            "restart",
            "hot_patch",
            "shutdown",
            "history",
        },
        "lifecycle",
    )
    lifecycle["soak_duration_s"] = _finite(
        lifecycle["soak_duration_s"], "lifecycle soak duration", maximum=604_800.0
    )
    raw_events = _strict_sequence(lifecycle["events"], "lifecycle events", maximum=16)
    events: list[dict[str, object]] = []
    for index, raw in enumerate(raw_events):
        event = _strict_mapping(
            raw, {"kind", "offset_s", "generation"}, f"event {index}"
        )
        if event["kind"] not in ("sustained", "restart", "hot_patch", "shutdown"):
            raise MattePlatformQualificationError("lifecycle event kind is invalid")
        event["offset_s"] = _finite(
            event["offset_s"], f"event {index} offset", maximum=604_800.0
        )
        event["generation"] = _exact_int(
            event["generation"],
            f"event {index} generation",
            minimum=1,
            maximum=2**31 - 1,
        )
        events.append(event)
    if [event["kind"] for event in events] != [
        "sustained",
        "restart",
        "hot_patch",
        "shutdown",
    ]:
        raise MattePlatformQualificationError(
            "lifecycle must record sustained, restart, hot patch, and shutdown in order"
        )
    if any(
        cast(float, current["offset_s"]) <= cast(float, previous["offset_s"])
        for previous, current in zip(events, events[1:])
    ):
        raise MattePlatformQualificationError("lifecycle event offsets must increase")
    soak_duration = cast(float, lifecycle["soak_duration_s"])
    if (
        events[0]["offset_s"] != 0.0
        or any(cast(float, event["offset_s"]) > soak_duration for event in events[:-1])
        or cast(float, events[-1]["offset_s"]) < soak_duration
        or cast(float, events[-1]["offset_s"]) > soak_duration + 120.0
    ):
        raise MattePlatformQualificationError(
            "lifecycle transitions are not bound to the declared soak window"
        )
    generations = [cast(int, event["generation"]) for event in events]
    if not (
        generations[1] == generations[0] + 1
        and generations[2] == generations[1] + 1
        and generations[3] == generations[2]
        and cast(float, events[1]["offset_s"]) - cast(float, events[0]["offset_s"])
        >= 60.0
        and cast(float, events[2]["offset_s"]) - cast(float, events[1]["offset_s"])
        >= 60.0
        and cast(float, events[3]["offset_s"]) - cast(float, events[2]["offset_s"])
        >= 60.0
    ):
        raise MattePlatformQualificationError(
            "restart/hot-patch generation transitions are invalid"
        )

    counter_names = (
        "unique_capture_frames",
        "unique_segmentations",
        "unique_composites",
        "sink_submissions",
        "capture_gaps",
        "capture_drops",
        "capture_slot_overwrites",
        "processing_deadline_misses",
        "sink_recoveries",
    )
    marker_kinds = (
        "sustained_start",
        "pre_restart",
        "post_restart",
        "pre_hot_patch",
        "post_hot_patch",
        "shutdown",
    )
    raw_snapshots = _strict_sequence(
        lifecycle["counter_snapshots"],
        "lifecycle counter snapshots",
        maximum=2048,
    )
    if len(raw_snapshots) < MIN_RESOURCE_SAMPLES:
        raise MattePlatformQualificationError(
            "lifecycle counter snapshots are too sparse"
        )
    snapshots: list[dict[str, object]] = []
    snapshot_keys = {
        "kind",
        "offset_s",
        "generation",
        "active_provider",
        *counter_names,
    }
    for index, raw in enumerate(raw_snapshots):
        snapshot = _strict_mapping(
            raw, snapshot_keys, f"lifecycle counter snapshot {index}"
        )
        if snapshot["kind"] not in (*marker_kinds, "heartbeat"):
            raise MattePlatformQualificationError(
                "lifecycle counter snapshot kind is invalid"
            )
        snapshot["offset_s"] = _finite(
            snapshot["offset_s"],
            f"lifecycle counter snapshot {index} offset",
            maximum=604_800.0,
        )
        snapshot["generation"] = _exact_int(
            snapshot["generation"],
            f"lifecycle counter snapshot {index} generation",
            minimum=1,
            maximum=2**31 - 1,
        )
        if snapshot["active_provider"] != provider:
            raise MattePlatformQualificationError(
                "lifecycle steady snapshots use the wrong active provider"
            )
        for name in counter_names:
            snapshot[name] = _exact_int(
                snapshot[name],
                f"lifecycle counter snapshot {index} {name}",
                minimum=0,
                maximum=2**63 - 1,
            )
        snapshots.append(snapshot)
    observed_markers = [
        cast(str, row["kind"]) for row in snapshots if row["kind"] != "heartbeat"
    ]
    if tuple(observed_markers) != marker_kinds:
        raise MattePlatformQualificationError(
            "lifecycle counter markers are incomplete or out of order"
        )
    marker_indexes = {
        cast(str, row["kind"]): index
        for index, row in enumerate(snapshots)
        if row["kind"] != "heartbeat"
    }
    snapshot_offsets = [cast(float, row["offset_s"]) for row in snapshots]
    maximum_snapshot_gap = max(
        current - previous
        for previous, current in zip(snapshot_offsets, snapshot_offsets[1:])
    )
    pre_restart_index = marker_indexes["pre_restart"]
    post_restart_index = marker_indexes["post_restart"]
    pre_hot_patch_index = marker_indexes["pre_hot_patch"]
    post_hot_patch_index = marker_indexes["post_hot_patch"]
    if (
        marker_indexes["sustained_start"] != 0
        or marker_indexes["shutdown"] != len(snapshots) - 1
        or snapshot_offsets[0] != 0.0
        or any(
            current <= previous
            for previous, current in zip(snapshot_offsets, snapshot_offsets[1:])
        )
        or snapshot_offsets[-1] != soak_duration
        or maximum_snapshot_gap > MAX_RESOURCE_SAMPLE_GAP_S
        or not snapshot_offsets[pre_restart_index]
        < cast(float, events[1]["offset_s"])
        < snapshot_offsets[post_restart_index]
        or not snapshot_offsets[pre_hot_patch_index]
        < cast(float, events[2]["offset_s"])
        < snapshot_offsets[post_hot_patch_index]
    ):
        raise MattePlatformQualificationError(
            "lifecycle snapshots do not bracket restart/hot-patch events"
        )
    restart_offset = cast(float, events[1]["offset_s"])
    hot_patch_offset = cast(float, events[2]["offset_s"])
    if any(
        row["generation"]
        != (
            generations[0]
            if cast(float, row["offset_s"]) < restart_offset
            else generations[1]
            if cast(float, row["offset_s"]) < hot_patch_offset
            else generations[2]
        )
        for row in snapshots
    ):
        raise MattePlatformQualificationError(
            "lifecycle counter generations do not match transition events"
        )
    if any(cast(int, snapshots[0][name]) != 0 for name in counter_names):
        raise MattePlatformQualificationError(
            "lifecycle counters must retain a zero origin"
        )
    interval_rates: dict[str, list[float]] = {name: [] for name in counter_names[:4]}
    interval_event_ratios: dict[str, list[float]] = {
        name: [] for name in counter_names[4:]
    }
    for previous, current in zip(snapshots, snapshots[1:]):
        if any(
            cast(int, current[name]) < cast(int, previous[name])
            for name in counter_names
        ):
            raise MattePlatformQualificationError(
                "lifecycle counters regress across the soak"
            )
        elapsed = cast(float, current["offset_s"]) - cast(float, previous["offset_s"])
        for name in counter_names[:4]:
            interval_rates[name].append(
                (cast(int, current[name]) - cast(int, previous[name])) / elapsed
            )
        capture_delta = cast(int, current["unique_capture_frames"]) - cast(
            int, previous["unique_capture_frames"]
        )
        composite_delta = cast(int, current["unique_composites"]) - cast(
            int, previous["unique_composites"]
        )
        for name in ("capture_gaps", "capture_drops", "capture_slot_overwrites"):
            event_delta = cast(int, current[name]) - cast(int, previous[name])
            interval_event_ratios[name].append(event_delta / max(1, capture_delta))
        deadline_delta = cast(int, current["processing_deadline_misses"]) - cast(
            int, previous["processing_deadline_misses"]
        )
        interval_event_ratios["processing_deadline_misses"].append(
            deadline_delta / max(1, composite_delta)
        )
        recovery_delta = cast(int, current["sink_recoveries"]) - cast(
            int, previous["sink_recoveries"]
        )
        interval_event_ratios["sink_recoveries"].append(float(recovery_delta))
    for before, after in (
        (snapshots[pre_restart_index], snapshots[post_restart_index]),
        (snapshots[pre_hot_patch_index], snapshots[post_hot_patch_index]),
    ):
        if cast(int, after["unique_composites"]) <= cast(
            int, before["unique_composites"]
        ) or cast(int, after["sink_submissions"]) <= cast(
            int, before["sink_submissions"]
        ):
            raise MattePlatformQualificationError(
                "lifecycle first post-transition output is not represented"
            )
    soak_rates = {
        name: round(
            (cast(int, snapshots[-1][name]) - cast(int, snapshots[0][name]))
            / soak_duration,
            6,
        )
        for name in counter_names[:4]
    }
    soak_event_counts = {
        name: cast(int, snapshots[-1][name]) - cast(int, snapshots[0][name])
        for name in counter_names[4:]
    }
    minimum_interval_rates = {
        name: round(min(values), 6) for name, values in interval_rates.items()
    }
    maximum_interval_event_ratios = {
        name: round(max(values), 6) for name, values in interval_event_ratios.items()
    }

    restart = _strict_mapping(
        lifecycle["restart"],
        {
            "reset_visible",
            "fallback_occurred",
            "fallback_visible",
            "first_output_fresh",
            "stale_state_flash",
            "provider_before",
            "provider_during_fallback",
            "provider_after_recovery",
            "recovery_generation",
            "first_output_generation",
        },
        "restart lifecycle",
    )
    for key in (
        "reset_visible",
        "fallback_occurred",
        "fallback_visible",
        "first_output_fresh",
        "stale_state_flash",
    ):
        if type(restart[key]) is not bool:
            raise MattePlatformQualificationError(f"restart.{key} must be boolean")
    if restart["fallback_visible"] != restart["fallback_occurred"]:
        raise MattePlatformQualificationError(
            "provider fallback visibility is inconsistent"
        )
    restart["recovery_generation"] = _exact_int(
        restart["recovery_generation"],
        "restart recovery generation",
        minimum=1,
        maximum=2**31 - 1,
    )
    restart["first_output_generation"] = _exact_int(
        restart["first_output_generation"],
        "restart first-output generation",
        minimum=1,
        maximum=2**31 - 1,
    )
    expected_fallback = provider != "cpu"
    if (
        restart["fallback_occurred"] is not expected_fallback
        or restart["provider_before"] != provider
        or restart["provider_after_recovery"] != provider
        or restart["provider_during_fallback"] != ("cpu" if expected_fallback else None)
        or restart["recovery_generation"] != generations[1]
        or restart["first_output_generation"] != generations[1]
    ):
        raise MattePlatformQualificationError(
            "runtime provider restart/fallback recovery is not exact"
        )

    hot_patch = _strict_mapping(
        lifecycle["hot_patch"],
        {
            "transactional",
            "reset_visible",
            "first_output_fresh",
            "stale_state_flash",
            "first_output_generation",
        },
        "hot-patch lifecycle",
    )
    for key in (
        "transactional",
        "reset_visible",
        "first_output_fresh",
        "stale_state_flash",
    ):
        if type(hot_patch[key]) is not bool:
            raise MattePlatformQualificationError(f"hot_patch.{key} must be boolean")
    hot_patch["first_output_generation"] = _exact_int(
        hot_patch["first_output_generation"],
        "hot-patch first-output generation",
        minimum=1,
        maximum=2**31 - 1,
    )
    if hot_patch["first_output_generation"] != generations[2]:
        raise MattePlatformQualificationError(
            "hot-patch first output is not bound to the new generation"
        )

    shutdown = _strict_mapping(
        lifecycle["shutdown"],
        {"duration_ms", "workers_alive", "resources_open", "close_error"},
        "shutdown lifecycle",
    )
    shutdown["duration_ms"] = _finite(
        shutdown["duration_ms"], "shutdown duration", maximum=60_000.0
    )
    shutdown["workers_alive"] = _exact_int(
        shutdown["workers_alive"], "shutdown workers", minimum=0, maximum=10_000
    )
    shutdown["resources_open"] = _exact_int(
        shutdown["resources_open"], "shutdown resources", minimum=0, maximum=10_000
    )
    if shutdown["close_error"] is not None:
        shutdown["close_error"] = _safe_id(
            shutdown["close_error"], "shutdown close error"
        )

    history = _strict_mapping(
        lifecycle["history"],
        {
            "bounded",
            "previous_frame_slots",
            "previous_mask_slots",
            "work_buffer_slots",
        },
        "temporal history",
    )
    if type(history["bounded"]) is not bool:
        raise MattePlatformQualificationError(
            "temporal history bounded must be boolean"
        )
    history["previous_frame_slots"] = _exact_int(
        history["previous_frame_slots"],
        "temporal previous-frame slots",
        minimum=0,
        maximum=1_000,
    )
    history["previous_mask_slots"] = _exact_int(
        history["previous_mask_slots"],
        "temporal previous-mask slots",
        minimum=0,
        maximum=1_000,
    )
    history["work_buffer_slots"] = _exact_int(
        history["work_buffer_slots"],
        "temporal work-buffer slots",
        minimum=0,
        maximum=100_000,
    )
    return {
        "soak_duration_s": lifecycle["soak_duration_s"],
        "events": events,
        "counter_snapshots": snapshots,
        "soak_rates": soak_rates,
        "minimum_interval_rates": minimum_interval_rates,
        "maximum_interval_event_ratios": maximum_interval_event_ratios,
        "soak_event_counts": soak_event_counts,
        "restart": restart,
        "hot_patch": hot_patch,
        "shutdown": shutdown,
        "history": history,
    }


def _gate(
    gate_id: str,
    passed: bool,
    *,
    value: object,
    limit: object,
    applicable: bool = True,
) -> dict[str, object]:
    return {
        "id": gate_id,
        "applicable": applicable,
        "passed": bool(passed) if applicable else None,
        "value": value,
        "limit": limit,
    }


def _evaluate_run(
    run: Mapping[str, object],
    *,
    cell: Mapping[str, object],
    capture: Mapping[str, object],
) -> tuple[list[dict[str, object]], str]:
    capture_outcome = cast(str, capture["outcome"])
    scope = cast(Mapping[str, object], run["scope"])
    warmups = cast(Mapping[str, int], scope["warmup_frames_by_source"])
    durations = cast(Mapping[str, float], scope["measured_seconds_by_source"])
    derived = cast(Mapping[str, Mapping[str, object]], run["derived"])
    gates: list[dict[str, object]] = []
    for source_name in ("physical_capture", "fixed_replay"):
        source = derived[source_name]
        counters = cast(Mapping[str, int], source["counters"])
        rates = cast(Mapping[str, float], source["rates"])
        timings = cast(Mapping[str, object], source["timings"])
        service = cast(Mapping[str, float], timings["complete_service_ms"])
        serialized = cast(Mapping[str, float], timings["serialized_cycle_ms"])
        compositor = cast(Mapping[str, float], timings["compositor_ms"])
        refinement = cast(Mapping[str, float], timings["refinement_ms"])
        output_interval = timings["output_interval_ms"]
        jitter = timings["inter_frame_jitter_ms"]
        lateness = cast(Mapping[str, float], timings["schedule_lateness_ms"])
        unique_capture_interval = timings["unique_capture_interval_ms"]
        unique_composite_interval = timings["unique_composite_interval_ms"]
        e2e_age = cast(Mapping[str, float], timings["end_to_end_age_ms"])
        output_max = (
            cast(Mapping[str, float], output_interval)["max"]
            if isinstance(output_interval, Mapping)
            else None
        )
        jitter_p95 = (
            cast(Mapping[str, float], jitter)["p95"]
            if isinstance(jitter, Mapping)
            else None
        )
        unique_capture_max = (
            cast(Mapping[str, float], unique_capture_interval)["max"]
            if isinstance(unique_capture_interval, Mapping)
            else None
        )
        unique_composite_max = (
            cast(Mapping[str, float], unique_composite_interval)["max"]
            if isinstance(unique_composite_interval, Mapping)
            else None
        )
        frame_ms = 1000.0 / TARGET_FPS
        gates.extend(
            (
                _gate(
                    f"{source_name}-warmup",
                    warmups[source_name] >= MIN_WARMUP_FRAMES,
                    value=warmups[source_name],
                    limit=f">={MIN_WARMUP_FRAMES}",
                ),
                _gate(
                    f"{source_name}-sample-count",
                    counters["measured_frames"] >= MIN_MEASURED_FRAMES,
                    value=counters["measured_frames"],
                    limit=f">={MIN_MEASURED_FRAMES}",
                ),
                _gate(
                    f"{source_name}-duration",
                    durations[source_name] >= MIN_MEASURED_SECONDS,
                    value=durations[source_name],
                    limit=f">={MIN_MEASURED_SECONDS}",
                ),
                _gate(
                    f"{source_name}-capture-rate",
                    rates["capture_fps"]
                    >= TARGET_FPS * MIN_CAPTURE_AND_OUTPUT_TARGET_RATIO,
                    value=rates["capture_fps"],
                    limit=f">={TARGET_FPS * MIN_CAPTURE_AND_OUTPUT_TARGET_RATIO}",
                ),
                _gate(
                    f"{source_name}-segmentation-rate",
                    rates["segmentation_fps"] >= MIN_UNIQUE_COMPOSITES_PER_S,
                    value=rates["segmentation_fps"],
                    limit=f">={MIN_UNIQUE_COMPOSITES_PER_S}",
                ),
                _gate(
                    f"{source_name}-unique-composite-rate",
                    rates["unique_composite_fps"] >= MIN_UNIQUE_COMPOSITES_PER_S,
                    value=rates["unique_composite_fps"],
                    limit=f">={MIN_UNIQUE_COMPOSITES_PER_S}",
                ),
                _gate(
                    f"{source_name}-output-rate",
                    rates["output_fps"]
                    >= TARGET_FPS * MIN_CAPTURE_AND_OUTPUT_TARGET_RATIO,
                    value=rates["output_fps"],
                    limit=f">={TARGET_FPS * MIN_CAPTURE_AND_OUTPUT_TARGET_RATIO}",
                ),
                _gate(
                    f"{source_name}-complete-service-p95",
                    service["p95"] <= MAX_COMPLETE_SERVICE_P95_MS,
                    value=service["p95"],
                    limit=f"<={MAX_COMPLETE_SERVICE_P95_MS}",
                ),
                _gate(
                    f"{source_name}-serialized-cycle-p95",
                    serialized["p95"] <= MAX_COMPLETE_SERVICE_P95_MS,
                    value=serialized["p95"],
                    limit=f"<={MAX_COMPLETE_SERVICE_P95_MS}",
                ),
                _gate(
                    f"{source_name}-end-to-end-age-drift",
                    float(cast(float, source["e2e_age_p95_drift_ms"]))
                    <= MAX_E2E_AGE_DRIFT_MS,
                    value=source["e2e_age_p95_drift_ms"],
                    limit=f"<={MAX_E2E_AGE_DRIFT_MS}",
                ),
                _gate(
                    f"{source_name}-deadline-miss-ratio",
                    counters["processing_deadline_misses"]
                    <= math.floor(counters["measured_frames"] * 0.05),
                    value=(
                        counters["processing_deadline_misses"]
                        / max(1, counters["measured_frames"])
                    ),
                    limit="<=0.05",
                ),
                _gate(
                    f"{source_name}-sink-recovery-free-steady-window",
                    counters["sink_recoveries"] == 0,
                    value=counters["sink_recoveries"],
                    limit="==0",
                ),
                _gate(
                    f"{source_name}-output-maximum-gap",
                    output_max is not None and output_max <= 1.5 * frame_ms,
                    value=output_max,
                    limit=f"<={1.5 * frame_ms}",
                ),
                _gate(
                    f"{source_name}-output-jitter-p95",
                    jitter_p95 is not None and jitter_p95 <= 0.5 * frame_ms,
                    value=jitter_p95,
                    limit=f"<={0.5 * frame_ms}",
                ),
                _gate(
                    f"{source_name}-schedule-lateness-p95",
                    lateness["p95"] <= 1000.0 / TARGET_FPS,
                    value=lateness["p95"],
                    limit=f"<={1000.0 / TARGET_FPS}",
                ),
                _gate(
                    f"{source_name}-unique-capture-maximum-gap",
                    unique_capture_max is not None
                    and unique_capture_max <= 2.0 * frame_ms,
                    value=unique_capture_max,
                    limit=f"<={2.0 * frame_ms}",
                ),
                _gate(
                    f"{source_name}-unique-composite-maximum-gap",
                    unique_composite_max is not None
                    and unique_composite_max <= 2.0 * frame_ms,
                    value=unique_composite_max,
                    limit=f"<={2.0 * frame_ms}",
                ),
                _gate(
                    f"{source_name}-absolute-end-to-end-age",
                    e2e_age["p95"] <= 2.0 * frame_ms
                    and e2e_age["max"] <= 2.0 * frame_ms,
                    value={"p95": e2e_age["p95"], "max": e2e_age["max"]},
                    limit=f"p95,max<={2.0 * frame_ms}",
                ),
                _gate(
                    f"{source_name}-capture-gap-ratio",
                    counters["capture_gaps"]
                    <= math.floor(counters["measured_frames"] * 0.05),
                    value=counters["capture_gaps"]
                    / max(1, counters["measured_frames"]),
                    limit="<=0.05",
                ),
                _gate(
                    f"{source_name}-capture-drop-ratio",
                    counters["capture_drops"]
                    <= math.floor(counters["measured_frames"] * 0.05),
                    value=counters["capture_drops"]
                    / max(1, counters["measured_frames"]),
                    limit="<=0.05",
                ),
                _gate(
                    f"{source_name}-capture-slot-overwrite-ratio",
                    counters["capture_slot_overwrites"]
                    <= math.floor(counters["measured_frames"] * 0.05),
                    value=counters["capture_slot_overwrites"]
                    / max(1, counters["measured_frames"]),
                    limit="<=0.05",
                ),
            )
        )
        is_720p = cell["canvas"] == {"width": 1280, "height": 720}
        gates.extend(
            (
                _gate(
                    f"{source_name}-compositor-p95-720p",
                    compositor["p95"] <= MAX_COMPOSITOR_P95_720P_MS,
                    value=compositor["p95"],
                    limit=f"<={MAX_COMPOSITOR_P95_720P_MS}",
                    applicable=is_720p,
                ),
                _gate(
                    f"{source_name}-refinement-p95-720p",
                    refinement["p95"] <= MAX_REFINEMENT_P95_720P_MS,
                    value=refinement["p95"],
                    limit=f"<={MAX_REFINEMENT_P95_720P_MS}",
                    applicable=is_720p,
                ),
            )
        )

    physical_rate = cast(
        Mapping[str, float],
        cast(Mapping[str, object], derived["physical_capture"])["rates"],
    )["capture_fps"]
    capture_only_rate = float(cast(float, capture["availability_fps"]))
    gates.append(
        _gate(
            "physical-run-capture-rate-agrees-with-capture-only",
            abs(physical_rate - capture_only_rate) <= TARGET_FPS * 0.10,
            value={
                "full_run_fps": physical_rate,
                "capture_only_fps": capture_only_rate,
            },
            limit=f"absolute-delta<={TARGET_FPS * 0.10}",
        )
    )

    resource = cast(Mapping[str, object], run["resource"])
    rss = cast(Mapping[str, float], resource["rss_bytes"])
    vram = resource["vram_bytes"]
    lifecycle = cast(Mapping[str, object], run["lifecycle"])
    restart = cast(Mapping[str, object], lifecycle["restart"])
    hot_patch = cast(Mapping[str, object], lifecycle["hot_patch"])
    shutdown = cast(Mapping[str, object], lifecycle["shutdown"])
    history = cast(Mapping[str, object], lifecycle["history"])
    soak_rates = cast(Mapping[str, float], lifecycle["soak_rates"])
    minimum_interval_rates = cast(
        Mapping[str, float], lifecycle["minimum_interval_rates"]
    )
    soak_events = cast(Mapping[str, int], lifecycle["soak_event_counts"])
    maximum_interval_event_ratios = cast(
        Mapping[str, float], lifecycle["maximum_interval_event_ratios"]
    )
    physical = cast(Mapping[str, object], run["physical_authority"])
    frame_gpu_activity = (
        True
        if cast(Mapping[str, object], run["provider"])["name"] == "cpu"
        else all(
            cast(
                Mapping[str, float],
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], derived[source_name])["timings"],
                )["gpu_utilization_percent"],
            )["p50"]
            >= 1.0
            for source_name in ("physical_capture", "fixed_replay")
        )
    )
    gates.extend(
        (
            _gate(
                "capture-only-prerequisite",
                capture_outcome == "qualified",
                value=capture_outcome,
                limit="qualified",
            ),
            _gate(
                "resource-soak-sample-count",
                cast(int, resource["sample_count"]) >= MIN_RESOURCE_SAMPLES,
                value=resource["sample_count"],
                limit=f">={MIN_RESOURCE_SAMPLES}",
            ),
            _gate(
                "resource-soak-duration",
                cast(float, resource["span_s"]) >= MIN_SOAK_SECONDS
                and cast(float, lifecycle["soak_duration_s"]) >= MIN_SOAK_SECONDS
                and cast(float, resource["span_s"]) + 0.001
                >= cast(float, lifecycle["soak_duration_s"]),
                value={
                    "resource_span_s": resource["span_s"],
                    "lifecycle_soak_s": lifecycle["soak_duration_s"],
                },
                limit=f">={MIN_SOAK_SECONDS}",
            ),
            _gate(
                "resource-sample-maximum-gap",
                cast(float, resource["maximum_sample_gap_s"])
                <= MAX_RESOURCE_SAMPLE_GAP_S,
                value=resource["maximum_sample_gap_s"],
                limit=f"<={MAX_RESOURCE_SAMPLE_GAP_S}",
            ),
            _gate(
                "rss-drift",
                rss["signed_drift_bytes"] <= MAX_RSS_DRIFT_BYTES,
                value=rss["signed_drift_bytes"],
                limit=f"<={MAX_RSS_DRIFT_BYTES}",
            ),
            _gate(
                "rss-observed-span",
                rss["span_bytes"] <= MAX_RSS_SPAN_BYTES,
                value=rss["span_bytes"],
                limit=f"<={MAX_RSS_SPAN_BYTES}",
            ),
            _gate(
                "vram-drift",
                (
                    cast(Mapping[str, float], vram)["signed_drift_bytes"]
                    <= MAX_VRAM_DRIFT_BYTES
                    if isinstance(vram, Mapping)
                    else True
                ),
                value=(
                    cast(Mapping[str, object], vram)["signed_drift_bytes"]
                    if isinstance(vram, Mapping)
                    else None
                ),
                limit=f"<={MAX_VRAM_DRIFT_BYTES}",
                applicable=isinstance(vram, Mapping),
            ),
            _gate(
                "vram-observed-span",
                (
                    cast(Mapping[str, float], vram)["span_bytes"] <= MAX_VRAM_SPAN_BYTES
                    if isinstance(vram, Mapping)
                    else True
                ),
                value=(
                    cast(Mapping[str, object], vram)["span_bytes"]
                    if isinstance(vram, Mapping)
                    else None
                ),
                limit=f"<={MAX_VRAM_SPAN_BYTES}",
                applicable=isinstance(vram, Mapping),
            ),
            _gate(
                "accelerator-activity-observed",
                (
                    cast(Mapping[str, float], resource["gpu_percent"])["p50"] >= 1.0
                    and frame_gpu_activity
                    if isinstance(resource["gpu_percent"], Mapping)
                    else True
                ),
                value=(
                    cast(Mapping[str, object], resource["gpu_percent"])["p50"]
                    if isinstance(resource["gpu_percent"], Mapping)
                    else None
                ),
                limit=">=1% median resource and measured-frame activity",
                applicable=isinstance(resource["gpu_percent"], Mapping),
            ),
            _gate(
                "process-activity-observed",
                cast(Mapping[str, float], resource["process_cpu_percent"])["p50"] > 0.0,
                value=cast(Mapping[str, object], resource["process_cpu_percent"])[
                    "p50"
                ],
                limit=">0% median process CPU",
            ),
            _gate(
                "sustained-soak-throughput",
                all(
                    soak_rates[name] >= MIN_UNIQUE_COMPOSITES_PER_S
                    and minimum_interval_rates[name] >= MIN_UNIQUE_COMPOSITES_PER_S
                    for name in (
                        "unique_capture_frames",
                        "unique_segmentations",
                        "unique_composites",
                        "sink_submissions",
                    )
                ),
                value={
                    "aggregate": dict(soak_rates),
                    "minimum_heartbeat_interval": dict(minimum_interval_rates),
                },
                limit=f"aggregate-and-every-heartbeat>={MIN_UNIQUE_COMPOSITES_PER_S}/s",
            ),
            _gate(
                "sustained-soak-event-bounds",
                soak_events["capture_gaps"]
                <= math.floor(
                    soak_rates["unique_capture_frames"]
                    * cast(float, lifecycle["soak_duration_s"])
                    * 0.05
                )
                and soak_events["capture_drops"]
                <= math.floor(
                    soak_rates["unique_capture_frames"]
                    * cast(float, lifecycle["soak_duration_s"])
                    * 0.05
                )
                and soak_events["capture_slot_overwrites"]
                <= math.floor(
                    soak_rates["unique_capture_frames"]
                    * cast(float, lifecycle["soak_duration_s"])
                    * 0.05
                )
                and soak_events["processing_deadline_misses"]
                <= math.floor(
                    soak_rates["unique_composites"]
                    * cast(float, lifecycle["soak_duration_s"])
                    * 0.05
                )
                and soak_events["sink_recoveries"] == 0
                and all(
                    maximum_interval_event_ratios[name] <= 0.05
                    for name in (
                        "capture_gaps",
                        "capture_drops",
                        "capture_slot_overwrites",
                        "processing_deadline_misses",
                    )
                )
                and maximum_interval_event_ratios["sink_recoveries"] == 0.0,
                value={
                    "aggregate": dict(soak_events),
                    "maximum_heartbeat_ratio": dict(maximum_interval_event_ratios),
                },
                limit=(
                    "aggregate-and-every-heartbeat-gap/drop/overwrite/deadline"
                    "<=5%;sink-recovery==0"
                ),
            ),
            _gate(
                "runtime-provider-fallback-exercised",
                restart["fallback_occurred"] is True
                and restart["fallback_visible"] is True,
                value={
                    "occurred": restart["fallback_occurred"],
                    "visible": restart["fallback_visible"],
                },
                limit="runtime fallback occurred and was visible",
                applicable=cast(Mapping[str, object], run["provider"])["name"] != "cpu",
            ),
            _gate(
                "restart-reset-and-fresh-output",
                restart["reset_visible"] is True
                and restart["first_output_fresh"] is True
                and restart["stale_state_flash"] is False,
                value=restart,
                limit="visible-reset,fresh-first-output,no-stale-flash",
            ),
            _gate(
                "hot-patch-transaction-and-fresh-output",
                hot_patch["transactional"] is True
                and hot_patch["reset_visible"] is True
                and hot_patch["first_output_fresh"] is True
                and hot_patch["stale_state_flash"] is False,
                value=hot_patch,
                limit="transactional,visible-reset,fresh-first-output",
            ),
            _gate(
                "bounded-shutdown",
                cast(float, shutdown["duration_ms"]) <= MAX_SHUTDOWN_MS
                and shutdown["workers_alive"] == 0
                and shutdown["resources_open"] == 0
                and shutdown["close_error"] is None,
                value=shutdown,
                limit=f"<={MAX_SHUTDOWN_MS}ms,zero-workers,zero-resources",
            ),
            _gate(
                "bounded-temporal-history",
                history["bounded"] is True
                and cast(int, history["previous_frame_slots"]) <= 2
                and cast(int, history["previous_mask_slots"]) <= 2
                and cast(int, history["work_buffer_slots"]) <= 32,
                value=history,
                limit="<=2-frame,<=2-mask,<=32-work-buffers",
            ),
            _gate(
                "physical-capture-consumer-same-host-attestation",
                physical["capture_origin_attested"] is True
                and physical["consumer_origin_attested"] is True
                and physical["same_host_attested"] is True,
                value={
                    "capture": physical["capture_origin_attested"],
                    "consumer": physical["consumer_origin_attested"],
                    "same_host": physical["same_host_attested"],
                    "cryptographically_proven": physical["cryptographically_proven"],
                },
                limit="owner-attested; cryptographic proof is not claimed",
            ),
        )
    )
    failed = any(gate["applicable"] and gate["passed"] is False for gate in gates)
    physical_missing = not all(
        physical[key] is True
        for key in (
            "capture_origin_attested",
            "consumer_origin_attested",
            "same_host_attested",
        )
    )
    # Missing physical authority and an unattested capture are awaiting local
    # evidence.  A measured, attested gate violation is a rejection.
    non_authority_failures = [
        gate
        for gate in gates
        if gate["applicable"]
        and gate["passed"] is False
        and gate["id"]
        not in (
            "physical-capture-consumer-same-host-attestation",
            "capture-only-prerequisite",
        )
    ]
    if non_authority_failures or (capture_outcome == "failed" and not physical_missing):
        outcome = "failed"
    elif failed or physical_missing or capture_outcome != "qualified":
        outcome = "pending"
    else:
        outcome = "qualified"
    return gates, outcome


def qualify_plan(plan_path: Path | str) -> dict[str, Any]:
    """Validate and aggregate one owner-only MATTE-5.3 qualification plan."""

    plan, _path, plan_sha256 = _load_plan(plan_path)
    prerequisites, prerequisite_status, authorizations = _load_prerequisites(plan)
    profile_by_id = {
        cast(str, profile["id"]): profile
        for profile in cast(Sequence[Mapping[str, object]], plan["profiles"])
    }
    cell_results: list[dict[str, object]] = []
    for cell in cast(Sequence[Mapping[str, object]], plan["cells"]):
        if cell["state"] == "unavailable":
            cell_results.append(
                {
                    "id": cell["id"],
                    "route_id": cell["route_id"],
                    "profile_id": cell["profile_id"],
                    "provider": cell["provider"],
                    "canvas": cell["canvas"],
                    "target_fps": cell["target_fps"],
                    "dependency_profile": cell["dependency_profile"],
                    "outcome": "unavailable",
                    "reason": cell["reason"],
                    "capture": None,
                    "run": None,
                    "gates": [],
                }
            )
            continue
        capture, _capture_outcome, _capture_passed = _capture_report(
            cast(Mapping[str, object], cell["capture_report"]), cell
        )
        run = _load_run(
            cast(Mapping[str, object], cell["run"]),
            plan=plan,
            cell=cell,
            profile=profile_by_id[cast(str, cell["profile_id"])],
            capture=capture,
            authorizations=authorizations,
        )
        gates, outcome = _evaluate_run(run, cell=cell, capture=capture)
        provenance_kind = cast(Mapping[str, object], run["provenance"])["kind"]
        if provenance_kind == "generated" and outcome == "qualified":
            outcome = "pending"
        derived = cast(Mapping[str, Mapping[str, object]], run["derived"])
        cell_results.append(
            {
                "id": cell["id"],
                "route_id": cell["route_id"],
                "profile_id": cell["profile_id"],
                "provider": cell["provider"],
                "canvas": cell["canvas"],
                "target_fps": cell["target_fps"],
                "dependency_profile": cell["dependency_profile"],
                "outcome": outcome,
                "reason": None,
                "capture": capture,
                "run": {
                    "file_sha256": run["file_sha256"],
                    "evidence_sha256": run["evidence_sha256"],
                    "build_sha256": cast(Mapping[str, object], run["provenance"])[
                        "build_sha256"
                    ],
                    "model": run["model"],
                    "hardware": run["hardware"],
                    "runtime": run["runtime"],
                    "provider": {
                        key: cast(Mapping[str, object], run["provider"])[key]
                        for key in ("name", "device_id", "execution_verified")
                    },
                    "sink": run["sink"],
                    "selection": run["selection"],
                    "dependencies": {
                        key: cast(Mapping[str, object], run["dependencies"])[key]
                        for key in (
                            "profile",
                            "mediapipe_installed",
                            "gpu_provider_installed",
                            "installed_distribution_sha256",
                            "available_providers",
                            "mediapipe_probe",
                            "gpu_provider_probe",
                        )
                    },
                    "physical_authority": run["physical_authority"],
                    "scope": run["scope"],
                    "sources": {
                        "physical_capture": run["sources"]["physical_capture"],
                        "fixed_replay": {
                            key: run["sources"]["fixed_replay"][key]
                            for key in (
                                "kind",
                                "acquisition",
                                "source_sha256",
                                "bundle_manifest_sha256",
                                "measured_frame_lineage_sha256",
                                "warmup_frame_count",
                                "measured_frame_count",
                                "trace_sha256",
                            )
                        },
                    },
                    "measurements": {
                        source_name: derived[source_name]
                        for source_name in ("physical_capture", "fixed_replay")
                    },
                    "resource": run["resource"],
                    "lifecycle": run["lifecycle"],
                },
                "gates": gates,
            }
        )

    declared_profile_routes = {
        (cast(str, row["profile_id"]), cast(str, row["route_id"]))
        for row in cell_results
    }
    missing_profile_routes = [
        f"{profile_id}@{route_id}"
        for profile_id in PROFILE_CONTRACTS
        for route_id in ROUTE_CONTRACTS
        if (profile_id, route_id) not in declared_profile_routes
    ]
    missing_cpu_routes = [
        route_id
        for route_id in ROUTE_CONTRACTS
        if not any(
            row["route_id"] == route_id and row["provider"] == "cpu"
            for row in cell_results
        )
    ]
    required_lane_checks: list[tuple[str, bool]] = []
    for profile_id in PROFILE_CONTRACTS:
        for route_id in ROUTE_CONTRACTS:
            required_lane_checks.append(
                (
                    f"standard-cpu:{profile_id}@{route_id}",
                    any(
                        row["profile_id"] == profile_id
                        and row["route_id"] == route_id
                        and row["provider"] == "cpu"
                        and row["dependency_profile"] == "standard"
                        for row in cell_results
                    ),
                )
            )
    for provider, platform in (
        ("cuda", "linux"),
        ("cuda", "windows"),
        ("directml", "windows"),
    ):
        required_lane_checks.append(
            (
                f"rvm-{provider}@{platform}",
                any(
                    row["profile_id"] == "rvm_matting"
                    and row["provider"] == provider
                    and ROUTE_CONTRACTS[cast(str, row["route_id"])]["platform"]
                    == platform
                    and row["dependency_profile"] == "standard"
                    for row in cell_results
                ),
            )
        )
    for dependency in ("without_mediapipe", "without_gpu_provider"):
        for platform in ("linux", "darwin", "windows"):
            required_lane_checks.append(
                (
                    f"{dependency}@{platform}",
                    any(
                        row["dependency_profile"] == dependency
                        and ROUTE_CONTRACTS[cast(str, row["route_id"])]["platform"]
                        == platform
                        for row in cell_results
                    ),
                )
            )
    missing_required_lanes = [
        lane_id for lane_id, present in required_lane_checks if not present
    ]
    profile_results: list[dict[str, object]] = []
    for profile in cast(Sequence[Mapping[str, object]], plan["profiles"]):
        rows = [row for row in cell_results if row["profile_id"] == profile["id"]]
        missing_routes = [
            route_id
            for route_id in ROUTE_CONTRACTS
            if (cast(str, profile["id"]), route_id) not in declared_profile_routes
        ]
        missing_standard_cpu_routes = [
            route_id
            for route_id in ROUTE_CONTRACTS
            if not any(
                row["route_id"] == route_id
                and row["provider"] == "cpu"
                and row["dependency_profile"] == "standard"
                for row in rows
            )
        ]
        outcome = (
            "failed"
            if any(row["outcome"] == "failed" for row in rows)
            else (
                "qualified"
                if rows
                and not missing_routes
                and not missing_standard_cpu_routes
                and all(row["outcome"] == "qualified" for row in rows)
                else "pending"
            )
        )
        profile_results.append(
            {
                "id": profile["id"],
                "backend": profile["backend"],
                "backend_kind": profile["backend_kind"],
                "quality_claim": profile["quality_claim"],
                "outcome": outcome,
                "missing_routes": missing_routes,
                "missing_standard_cpu_routes": missing_standard_cpu_routes,
                "qualified_limits": [
                    row["id"] for row in rows if row["outcome"] == "qualified"
                ],
                "pending_limits": [
                    row["id"]
                    for row in rows
                    if row["outcome"] in ("pending", "unavailable")
                ],
                "failed_limits": [
                    row["id"] for row in rows if row["outcome"] == "failed"
                ],
            }
        )

    prerequisite_failed = (
        prerequisite_status["visual"] == "failed"
        or prerequisite_status["performance"] == "rejected"
        or any(row["outcome"] == "failed" for row in cell_results)
    )
    prerequisites_complete = prerequisite_status == {
        "visual": "qualified",
        "performance": "qualified",
        "rvm": "qualified",
    }
    provenance = cast(Mapping[str, object], plan["provenance"])
    authority_complete = (
        provenance["kind"] != "generated"
        and provenance["physical_capture_origin_attested"] is True
        and provenance["physical_consumer_origin_attested"] is True
        and provenance["same_host_attested"] is True
    )
    cells_complete = (
        bool(cell_results)
        and not missing_profile_routes
        and not missing_cpu_routes
        and not missing_required_lanes
        and all(row["outcome"] == "qualified" for row in cell_results)
    )
    if prerequisite_failed:
        status = "failed"
    elif prerequisites_complete and authority_complete and cells_complete:
        status = "qualified"
    else:
        status = "pending"
    reasons: list[str] = []
    if provenance["kind"] == "generated":
        reasons.append("generated evidence cannot qualify physical platforms")
    if not prerequisites_complete:
        reasons.append(
            "visual, fixed-replay performance, or RVM prerequisite is pending"
        )
    if not authority_complete:
        reasons.append(
            "local physical capture/consumer/same-host authority is incomplete"
        )
    unavailable = [row["id"] for row in cell_results if row["outcome"] == "unavailable"]
    pending = [row["id"] for row in cell_results if row["outcome"] == "pending"]
    failed = [row["id"] for row in cell_results if row["outcome"] == "failed"]
    if unavailable:
        reasons.append("required platform cells are explicitly unavailable")
    if pending:
        reasons.append("required platform cells await authoritative local evidence")
    if failed:
        reasons.append("recorded platform cells failed one or more code-owned gates")
    if missing_profile_routes:
        reasons.append("backend tiers silently omit reviewed platform routes")
    if missing_cpu_routes:
        reasons.append("reviewed platform routes silently omit CPU-only behavior")
    if missing_required_lanes:
        reasons.append("code-owned provider/dependency platform lanes are missing")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "status": status,
        "source": {"plan_sha256": plan_sha256},
        "provenance": {
            "qualification_id": cast(Mapping[str, object], plan["qualification"])["id"],
            "kind": provenance["kind"],
            "physical_origin_authority": "owner-attested",
            "physical_origin_cryptographically_proven": False,
            "candidate_build_authority": "owner-attested-exact-contract-join",
            "candidate_build_cryptographically_bound_to_prerequisites": False,
        },
        "privacy": {
            "report_is_path_free": True,
            "report_contains_pixels": False,
            "camera_model_preview_api_network_or_sink_opened": False,
            "private_inputs_owner_only": True,
        },
        "policy": {
            "target_fps": TARGET_FPS,
            "capture_v1_qualifying_canvas": {"width": 1280, "height": 720},
            "minimum_target_ratio": MIN_CAPTURE_AND_OUTPUT_TARGET_RATIO,
            "minimum_unique_composites_per_s": MIN_UNIQUE_COMPOSITES_PER_S,
            "complete_service_p95_ms": MAX_COMPLETE_SERVICE_P95_MS,
            "compositor_p95_720p_ms": MAX_COMPOSITOR_P95_720P_MS,
            "refinement_p95_720p_ms": MAX_REFINEMENT_P95_720P_MS,
            "minimum_warmup_frames_per_source": MIN_WARMUP_FRAMES,
            "minimum_measured_frames_per_source": MIN_MEASURED_FRAMES,
            "minimum_measured_seconds_per_source": MIN_MEASURED_SECONDS,
            "minimum_soak_seconds": MIN_SOAK_SECONDS,
            "maximum_shutdown_ms": MAX_SHUTDOWN_MS,
            "maximum_rss_drift_bytes": MAX_RSS_DRIFT_BYTES,
            "maximum_vram_drift_bytes": MAX_VRAM_DRIFT_BYTES,
            "maximum_rss_span_bytes": MAX_RSS_SPAN_BYTES,
            "maximum_vram_span_bytes": MAX_VRAM_SPAN_BYTES,
            "maximum_e2e_age_p95_drift_ms": MAX_E2E_AGE_DRIFT_MS,
            "percentile_method": "nearest-rank",
            "output_repeats_credited_as_unique_work": False,
        },
        "candidate": plan["candidate"],
        "prerequisites": prerequisites,
        "coverage": {
            "required_routes": list(ROUTE_CONTRACTS),
            "required_profiles": list(PROFILE_CONTRACTS),
            "required_providers": list(PROVIDERS),
            "required_dependency_profiles": list(DEPENDENCY_PROFILES),
            "required_profile_route_count": len(PROFILE_CONTRACTS)
            * len(ROUTE_CONTRACTS),
            "missing_profile_routes": missing_profile_routes,
            "missing_cpu_routes": missing_cpu_routes,
            "missing_required_lanes": missing_required_lanes,
            "declared_cell_count": len(cell_results),
            "qualified_cells": [
                row["id"] for row in cell_results if row["outcome"] == "qualified"
            ],
            "pending_cells": pending,
            "unavailable_cells": unavailable,
            "failed_cells": failed,
            "complete": cells_complete,
        },
        "routes": [
            {
                "id": route_id,
                **contract,
                "qualified_cells": [
                    row["id"]
                    for row in cell_results
                    if row["route_id"] == route_id and row["outcome"] == "qualified"
                ],
            }
            for route_id, contract in ROUTE_CONTRACTS.items()
        ],
        "profiles": profile_results,
        "cells": cell_results,
        "production": {
            "defaults_changed": False,
            "preset_catalog_changed": False,
            "generated_evidence_can_qualify": False,
            "reactions_enabled": False,
            "react_budget_qualified_here": False,
        },
        "reasons": reasons,
    }
    report["evidence_sha256"] = _report_digest(report)
    return report


def report_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Matte platform qualification report",
        "",
        f"- Status: **{report['status']}**",
        f"- Evidence SHA-256: `{report['evidence_sha256']}`",
        "- Physical origin: **owner-attested; not cryptographically proven**",
        "- Production defaults changed: **no**",
        "",
        "| Cell | Route | Profile | Provider | Dependency | Outcome |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cast(Sequence[Mapping[str, object]], report["cells"]):
        lines.append(
            f"| `{cell['id']}` | `{cell['route_id']}` | `{cell['profile_id']}` | "
            f"`{cell['provider']}` | `{cell['dependency_profile']}` | "
            f"**{cell['outcome']}** |"
        )
    lines.extend(
        (
            "",
            "Generated/proxy evidence and unavailable physical rows remain pending.",
            "",
        )
    )
    return "\n".join(lines)


def run_qualification(plan_path: Path | str, output_root: Path | str) -> dict[str, Any]:
    """Write a new owner-only path-free report directory."""

    report = qualify_plan(plan_path)
    output = Path(output_root)
    try:
        _private_directory(output, create=True)
        _atomic_private_write(output / "qualification.json", _json_bytes(report))
        _atomic_private_write(
            output / "qualification.md", report_markdown(report).encode("utf-8")
        )
    except MatteDiagnosticsError as exc:
        raise MattePlatformQualificationError(str(exc)) from exc
    return report


def build_parser(
    *, prog: str = "custback matte-platform-qualify"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate private MATTE-5.3 platform evidence offline",
    )
    parser.add_argument("plan", help="owner-only qualification plan")
    parser.add_argument(
        "--output", required=True, help="new owner-only content-free report directory"
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback matte-platform-qualify",
) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        report = run_qualification(args.plan, args.output)
    except (OSError, ValueError) as exc:
        print(f"{prog}: {str(exc).splitlines()[0]}", file=sys.stderr)
        return 2
    print(
        f"evaluated {len(report['cells'])} MATTE platform cell(s); "
        f"status {report['status']}"
    )
    return 0 if report["status"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
