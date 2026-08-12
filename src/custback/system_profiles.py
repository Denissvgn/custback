"""Immutable, evidence-gated system profile catalog and matching helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Mapping

from .config import AppConfig, config_lifecycle
from .vcam import SUPPORTED_NATIVE_MODES


CATALOG_RESOURCE = "system-profile-catalog.json"
CATALOG_SCHEMA = "custback.system-profile-catalog"
CATALOG_VERSION = 1
NONQUALIFIED_EVIDENCE_STATES = frozenset({"experimental", "locally_screened"})
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
_LEAF_PATH = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z")
_MAX_CATALOG_BYTES = 256 * 1024
_ALLOWED_DEFINITION_KEYS = frozenset(
    {
        "label",
        "description",
        "evidence_state",
        "selectable",
        "quality_claim",
        "requirements",
        "patch",
    }
)
_ALLOWED_REQUIREMENTS = frozenset({"builtin_rvm_model", "provider", "output_mode"})
_FORBIDDEN_PREFIXES = (
    "api.",
    "avatar.",
    "background.",
    "backdrop_targets.",
)
_FORBIDDEN_PATHS = frozenset(
    {
        "acceleration.device_id",
        "camera.device",
        "output.backend",
        "output.device",
        "segmentation.model_path",
    }
)


class ProfileCatalogError(ValueError):
    """The packaged profile catalog violated its reviewed contract."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def flatten_patch(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten a concrete nested patch, rejecting empty/intermediate leaves."""

    if not isinstance(value, dict) or not value:
        raise ProfileCatalogError("profile patches must be non-empty objects")
    flattened: dict[str, object] = {}
    for key in sorted(value):
        if not isinstance(key, str) or not _IDENTIFIER.fullmatch(key):
            raise ProfileCatalogError("profile patch contains an invalid field name")
        path = f"{prefix}.{key}" if prefix else key
        child = value[key]
        if isinstance(child, dict):
            flattened.update(flatten_patch(child, path))
        elif child is None:
            raise ProfileCatalogError(f"profile patch leaf {path} may not be null")
        else:
            flattened[path] = child
    return flattened


def get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def set_path(value: dict[str, Any], path: str, leaf: Any) -> None:
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ProfileCatalogError(f"profile path overlaps scalar field: {path}")
        current = child
    current[parts[-1]] = copy.deepcopy(leaf)


def delete_path(value: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    parents: list[tuple[dict[str, Any], str]] = []
    current = value
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        parents.append((current, part))
        current = child
    current.pop(parts[-1], None)
    for parent, part in reversed(parents):
        child = parent.get(part)
        if isinstance(child, dict) and not child:
            parent.pop(part, None)


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(child) for child in value)
    return value


def _thawed(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thawed(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thawed(child) for child in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ProfileDefinition:
    axis: str
    identifier: str
    label: str
    description: str
    evidence_state: str
    selectable: bool
    quality_claim: bool
    requirements: Mapping[str, Any]
    patch: Mapping[str, Any]
    owned_paths: tuple[str, ...]
    lifecycle: str
    patch_digest: str

    def detached_patch(self) -> dict[str, Any]:
        return _thawed(self.patch)


@dataclass(frozen=True)
class ProfileCatalog:
    version: int
    digest: str
    quality_claim: bool
    axes: Mapping[str, Mapping[str, ProfileDefinition]]
    axis_labels: Mapping[str, str]
    owned_paths: Mapping[str, tuple[str, ...]]

    def definition(self, axis: str, identifier: str) -> ProfileDefinition:
        try:
            return self.axes[axis][identifier]
        except KeyError:
            raise ProfileCatalogError(
                f"unknown profile selection: {axis}.{identifier}"
            ) from None

    @property
    def manageable_paths(self) -> frozenset[str]:
        return frozenset(path for paths in self.owned_paths.values() for path in paths)


def _validate_requirements(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not set(value).issubset(_ALLOWED_REQUIREMENTS):
        raise ProfileCatalogError("profile requirements contain unsupported fields")
    if "builtin_rvm_model" in value and value["builtin_rvm_model"] is not True:
        raise ProfileCatalogError("builtin_rvm_model requirement must be true")
    if "provider" in value and value["provider"] != "cuda":
        raise ProfileCatalogError("only the reviewed CUDA requirement is supported")
    if "output_mode" in value:
        mode = value["output_mode"]
        if (
            not isinstance(mode, list)
            or len(mode) != 3
            or any(type(item) is not int or item <= 0 for item in mode)
        ):
            raise ProfileCatalogError(
                "output_mode must contain three positive integers"
            )
    return copy.deepcopy(value)


def _validate_requirement_patch(
    requirements: Mapping[str, object],
    leaves: Mapping[str, object],
    *,
    profile: str,
) -> None:
    """Bind descriptive capability requirements to enforcing patch leaves."""

    if (
        requirements.get("builtin_rvm_model")
        and leaves.get("segmentation.backend") != "rvm"
    ):
        raise ProfileCatalogError(
            f"profile {profile} built-in RVM requirement is not enforced"
        )
    if requirements.get("provider") == "cuda" and (
        leaves.get("acceleration.mode") != "gpu_required"
        or leaves.get("acceleration.provider") != "cuda"
    ):
        raise ProfileCatalogError(f"profile {profile} CUDA requirement is not enforced")
    output_mode = requirements.get("output_mode")
    if output_mode is not None and (
        not isinstance(output_mode, (list, tuple))
        or tuple(output_mode)
        != (
            leaves.get("output.width"),
            leaves.get("output.height"),
            leaves.get("output.fps"),
        )
    ):
        raise ProfileCatalogError(
            f"profile {profile} output-mode requirement disagrees with its patch"
        )


def _load_catalog(encoded: bytes) -> ProfileCatalog:
    if len(encoded) > _MAX_CATALOG_BYTES:
        raise ProfileCatalogError("profile catalog exceeds its size limit")
    try:
        raw = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileCatalogError("profile catalog is not valid JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "version",
        "quality_claim",
        "axes",
    }:
        raise ProfileCatalogError("profile catalog root does not match schema")
    if raw["schema"] != CATALOG_SCHEMA or raw["version"] != CATALOG_VERSION:
        raise ProfileCatalogError("profile catalog schema/version is unsupported")
    if raw["quality_claim"] is not False:
        raise ProfileCatalogError("experimental catalog must not make a quality claim")
    axes_raw = raw["axes"]
    if not isinstance(axes_raw, dict) or not axes_raw:
        raise ProfileCatalogError("profile catalog must define axes")

    all_owned: set[str] = set()
    axes: dict[str, Mapping[str, ProfileDefinition]] = {}
    labels: dict[str, str] = {}
    owned_by_axis: dict[str, tuple[str, ...]] = {}
    base = AppConfig()
    for axis, axis_raw in axes_raw.items():
        if not isinstance(axis, str) or not _IDENTIFIER.fullmatch(axis):
            raise ProfileCatalogError("profile catalog contains an invalid axis ID")
        if not isinstance(axis_raw, dict) or set(axis_raw) != {
            "label",
            "owned_paths",
            "profiles",
        }:
            raise ProfileCatalogError(f"profile axis {axis} does not match schema")
        label = axis_raw["label"]
        owned_raw = axis_raw["owned_paths"]
        profiles_raw = axis_raw["profiles"]
        if not isinstance(label, str) or not 1 <= len(label) <= 80:
            raise ProfileCatalogError(f"profile axis {axis} has an invalid label")
        if (
            not isinstance(owned_raw, list)
            or not owned_raw
            or len(owned_raw) != len(set(owned_raw))
            or any(
                not isinstance(path, str) or not _LEAF_PATH.fullmatch(path)
                for path in owned_raw
            )
        ):
            raise ProfileCatalogError(f"profile axis {axis} has invalid owned paths")
        owned = tuple(sorted(owned_raw))
        for path in owned:
            if path in _FORBIDDEN_PATHS or path.startswith(_FORBIDDEN_PREFIXES):
                raise ProfileCatalogError(f"profile path is forbidden: {path}")
            if path in all_owned:
                raise ProfileCatalogError(f"profile axes overlap at {path}")
        all_owned.update(owned)
        if not isinstance(profiles_raw, dict) or not profiles_raw:
            raise ProfileCatalogError(f"profile axis {axis} has no definitions")

        definitions: dict[str, ProfileDefinition] = {}
        for identifier, definition_raw in profiles_raw.items():
            if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
                raise ProfileCatalogError(
                    "profile catalog contains an invalid profile ID"
                )
            if (
                not isinstance(definition_raw, dict)
                or set(definition_raw) != _ALLOWED_DEFINITION_KEYS
            ):
                raise ProfileCatalogError(
                    f"profile definition {axis}.{identifier} does not match schema"
                )
            label_value = definition_raw["label"]
            description = definition_raw["description"]
            if not isinstance(label_value, str) or not 1 <= len(label_value) <= 80:
                raise ProfileCatalogError(
                    f"profile {axis}.{identifier} has an invalid label"
                )
            if not isinstance(description, str) or not 1 <= len(description) <= 240:
                raise ProfileCatalogError(
                    f"profile {axis}.{identifier} has an invalid description"
                )
            evidence_state = definition_raw["evidence_state"]
            if evidence_state not in NONQUALIFIED_EVIDENCE_STATES:
                raise ProfileCatalogError(
                    "profiles must remain experimental or locally screened"
                )
            if (
                definition_raw["selectable"] is not True
                or definition_raw["quality_claim"] is not False
            ):
                raise ProfileCatalogError(
                    "non-qualified profiles must be selectable without a quality claim"
                )
            requirements = _validate_requirements(definition_raw["requirements"])
            patch = copy.deepcopy(definition_raw["patch"])
            leaves = flatten_patch(patch)
            if tuple(sorted(leaves)) != owned:
                raise ProfileCatalogError(
                    f"profile {axis}.{identifier} must own every axis leaf exactly"
                )
            _validate_requirement_patch(
                requirements,
                leaves,
                profile=f"{axis}.{identifier}",
            )
            candidate = base.patched(patch)
            lifecycle = config_lifecycle(base, candidate)
            if lifecycle != "restart":
                raise ProfileCatalogError(
                    f"initial profile {axis}.{identifier} must be restart-bound"
                )
            definitions[identifier] = ProfileDefinition(
                axis=axis,
                identifier=identifier,
                label=label_value,
                description=description,
                evidence_state=evidence_state,
                selectable=True,
                quality_claim=False,
                requirements=_frozen(requirements),
                patch=_frozen(patch),
                owned_paths=owned,
                lifecycle=lifecycle,
                patch_digest=_sha256(patch),
            )
        axes[axis] = MappingProxyType(definitions)
        labels[axis] = label
        owned_by_axis[axis] = owned
    return ProfileCatalog(
        version=CATALOG_VERSION,
        digest=hashlib.sha256(encoded).hexdigest(),
        quality_claim=False,
        axes=MappingProxyType(axes),
        axis_labels=MappingProxyType(labels),
        owned_paths=MappingProxyType(owned_by_axis),
    )


def load_profile_catalog(encoded: bytes | None = None) -> ProfileCatalog:
    """Load and validate the reviewed package resource (or test bytes)."""

    if encoded is None:
        encoded = files("custback").joinpath(CATALOG_RESOURCE).read_bytes()
    return _load_catalog(encoded)


PROFILE_CATALOG = load_profile_catalog()


def matches_profile(config: AppConfig, definition: ProfileDefinition) -> bool:
    values = config.to_dict()
    flattened = flatten_patch(definition.detached_patch())
    return all(
        get_path(values, path) == expected for path, expected in flattened.items()
    )


def matching_profile(
    config: AppConfig, axis: str, catalog: ProfileCatalog = PROFILE_CATALOG
) -> str | None:
    matches = [
        identifier
        for identifier, definition in catalog.axes[axis].items()
        if matches_profile(config, definition)
    ]
    if len(matches) > 1:
        raise ProfileCatalogError(f"ambiguous profile match for axis {axis}")
    return matches[0] if matches else None


def profile_availability(
    definition: ProfileDefinition,
    config: AppConfig,
    runtime_facts: Mapping[str, Any] | None = None,
    *,
    platform: str | None = None,
) -> tuple[bool, str]:
    """Evaluate bounded host requirements without exposing configuration paths."""

    facts = {} if runtime_facts is None else runtime_facts
    requirements = definition.requirements
    if requirements.get("builtin_rvm_model") and config.segmentation.model_path:
        return (
            False,
            "A custom segmentation model is configured; use the built-in RVM model.",
        )
    if requirements:
        if facts.get("output_fallback_active") is not False:
            return False, "The configured output sink is currently in fallback mode."
        if facts.get("segmentation_fallback_active") is not False:
            return False, "The active segmentation backend is a fallback."
        if facts.get("acceleration_fallback_active") is not False:
            return False, "CUDA execution is currently in fallback mode."
        selection = facts.get("segmentation_selection")
        if (
            not isinstance(selection, Mapping)
            or str(selection.get("selected_backend", "")).lower() != "rvm"
        ):
            return False, "Built-in RVM execution has not been proven for this run."
    if requirements.get("provider") == "cuda":
        selection = facts.get("segmentation_selection")
        active_provider = str(facts.get("acceleration_active_provider", "")).lower()
        active_device = str(facts.get("segmentation_device", "")).lower()
        selected_provider = (
            str(selection.get("active_provider", "")).lower()
            if isinstance(selection, Mapping)
            else ""
        )
        if (
            facts.get("acceleration_state") != "gpu_active"
            or active_provider not in {"cuda", "cudaexecutionprovider"}
            or selected_provider not in {"cuda", "cudaexecutionprovider"}
            or active_device != "cuda"
        ):
            return False, "CUDA execution has not been proven for this run."
    output_mode = requirements.get("output_mode")
    backend = str(facts.get("output_backend", "")).lower()
    current_platform = sys.platform if platform is None else platform
    if requirements and backend in {"null", "nulloutput"}:
        return False, "API-only output does not satisfy the profile sink requirement."
    if isinstance(output_mode, tuple):
        output_mode = list(output_mode)
    if output_mode and (
        backend in {"native", "nativevirtualcameraoutput"}
        or (
            not backend
            and current_platform == "win32"
            and config.output.backend == "native"
        )
    ):
        if tuple(output_mode) not in SUPPORTED_NATIVE_MODES:
            return (
                False,
                "The Windows native output does not support this exact canvas and frame rate.",
            )
    if output_mode:
        requested_mode = tuple(output_mode)
        supported_modes: set[tuple[object, ...]] = set()
        raw_supported = facts.get("supported_output_modes")
        if isinstance(raw_supported, (list, tuple)):
            for raw_mode in raw_supported:
                if isinstance(raw_mode, (list, tuple)) and len(raw_mode) == 3:
                    supported_modes.add(tuple(raw_mode))
        if backend in {"native", "nativevirtualcameraoutput"}:
            supported_modes.update(SUPPORTED_NATIVE_MODES)
        active_mode = None
        if all(key in facts for key in ("output_width", "output_height", "output_fps")):
            active_mode = (
                facts.get("output_width"),
                facts.get("output_height"),
                facts.get("output_fps"),
            )
        if requested_mode != active_mode and requested_mode not in supported_modes:
            return (
                False,
                "This exact output canvas and frame rate has not been proven for the active sink.",
            )
    if requirements and not backend:
        return False, "Runtime output capability is not available yet."
    return True, ""


def public_catalog(catalog: ProfileCatalog = PROFILE_CATALOG) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for axis, definitions in catalog.axes.items():
        axes[axis] = {
            "label": catalog.axis_labels[axis],
            "owned_paths": list(catalog.owned_paths[axis]),
            "profiles": {
                identifier: {
                    "id": identifier,
                    "label": definition.label,
                    "description": definition.description,
                    "evidence_state": definition.evidence_state,
                    "selectable": definition.selectable,
                    "quality_claim": definition.quality_claim,
                    "requirements": _thawed(definition.requirements),
                    "lifecycle": definition.lifecycle,
                    "patch_digest": definition.patch_digest,
                }
                for identifier, definition in definitions.items()
            },
        }
    return {
        "schema": CATALOG_SCHEMA,
        "version": catalog.version,
        "digest": catalog.digest,
        "quality_claim": catalog.quality_claim,
        "axes": axes,
    }


__all__ = [
    "CATALOG_SCHEMA",
    "CATALOG_VERSION",
    "NONQUALIFIED_EVIDENCE_STATES",
    "PROFILE_CATALOG",
    "ProfileCatalog",
    "ProfileCatalogError",
    "ProfileDefinition",
    "canonical_json",
    "delete_path",
    "flatten_patch",
    "get_path",
    "load_profile_catalog",
    "matches_profile",
    "matching_profile",
    "profile_availability",
    "public_catalog",
    "set_path",
]
