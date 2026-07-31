"""Deterministic offline metrics for private matte replay bundles.

This module never opens a camera, model, API, or virtual output.  It evaluates
lossless arrays from :mod:`custback.matte_diagnostics`, optionally joined to a
digest-bound private annotation bundle.  Metric definitions and release gates
remain report data rather than mutable production behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence, cast

import cv2
import numpy as np

from .matte_diagnostics import (
    MAX_ARTIFACT_BYTES,
    MAX_MANIFEST_BYTES,
    MatteDiagnosticsError,
    MatteReplayBundle,
    _atomic_private_write,
    _json_bytes,
    _npy_bytes,
    _private_directory,
    _private_write,
    _read_private_file,
)

ANNOTATION_SCHEMA = "custback.matte-quality-annotations"
ANNOTATION_VERSION = 1
REPORT_SCHEMA = "custback.matte-quality-report"
REPORT_VERSION = 1
POST_BASE_SCHEMA = "custback.matte-post-base-output-provenance"
METRIC_TOLERANCE = 1e-6

AnnotationKind = Literal[
    "opaque_core",
    "background",
    "ground_truth_alpha",
    "ground_truth_foreground",
]
RegionKind = Literal["opaque_core", "background", "soft_boundary"]
_ANNOTATION_KINDS: tuple[AnnotationKind, ...] = (
    "opaque_core",
    "background",
    "ground_truth_alpha",
    "ground_truth_foreground",
)
_REGION_KINDS: tuple[RegionKind, ...] = (
    "opaque_core",
    "background",
    "soft_boundary",
)
_REGION_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class MatteQualityError(ValueError):
    """A bundle, annotation, metric, or gate violates the quality contract."""


@dataclass(frozen=True)
class QualityNamedRegion:
    """One named spatial trimap region used for defect attribution."""

    kind: RegionKind
    mask: np.ndarray


@dataclass(frozen=True)
class QualityFrameAnnotations:
    """In-memory annotations for one unique replay input."""

    segment: str
    registration_from_previous: tuple[float, float, float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    opaque_core: np.ndarray | None = None
    background: np.ndarray | None = None
    ground_truth_alpha: np.ndarray | None = None
    ground_truth_foreground: np.ndarray | None = None
    regions: Mapping[str, QualityNamedRegion] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationMetadata:
    """Review metadata that cannot be inferred reliably from stored pixels."""

    hardware_label: str = ""
    backend: str = ""
    device: str = ""
    effective_detail: str = ""
    resampling: str = ""
    configuration_label: str = ""
    notes: str = ""


@dataclass
class _MetricSeries:
    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, metrics: Mapping[str, object]) -> None:
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            scalar = float(value)
            if math.isfinite(scalar):
                self.values.setdefault(name, []).append(scalar)

    def summaries(self) -> dict[str, dict[str, float | int | None]]:
        return {name: _summary(values) for name, values in sorted(self.values.items())}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 8)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": _round(float(np.mean(array, dtype=np.float64))),
        "p05": _round(float(np.percentile(array, 5))),
        "p50": _round(float(np.percentile(array, 50))),
        "p95": _round(float(np.percentile(array, 95))),
        "min": _round(float(np.min(array))),
        "max": _round(float(np.max(array))),
    }


def _safe_annotation_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise MatteQualityError("annotation artifact path is malformed")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "arrays"
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise MatteQualityError("annotation artifact path escapes the bundle")
    return relative


def _array_descriptor(path: str, payload: bytes, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _bundle_manifest_digest(bundle: MatteReplayBundle) -> str:
    payload = _read_private_file(
        bundle.root / "manifest.json",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    return _sha256(payload)


def _validated_annotation_array(
    value: np.ndarray | None,
    *,
    kind: AnnotationKind,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if kind in ("opaque_core", "background"):
        if array.ndim != 2 or array.shape != shape:
            raise MatteQualityError(f"{kind} annotation shape is invalid")
        return np.ascontiguousarray(array.astype(bool).astype(np.uint8))
    if kind == "ground_truth_alpha":
        if (
            array.dtype != np.float32
            or array.ndim != 2
            or array.shape != shape
            or not bool(np.isfinite(array).all())
            or float(np.min(array)) < 0.0
            or float(np.max(array)) > 1.0
        ):
            raise MatteQualityError(
                "ground_truth_alpha must be finite float32 alpha in [0, 1]"
            )
        return np.ascontiguousarray(array)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape != (*shape, 3):
        raise MatteQualityError("ground_truth_foreground must be matching uint8 BGR")
    return np.ascontiguousarray(array)


def _validated_region(
    name: object,
    value: object,
    *,
    shape: tuple[int, int],
) -> tuple[str, QualityNamedRegion]:
    if not isinstance(name, str) or _REGION_NAME.fullmatch(name) is None:
        raise MatteQualityError(
            "region names must be lowercase identifiers up to 64 characters"
        )
    if not isinstance(value, QualityNamedRegion) or value.kind not in _REGION_KINDS:
        raise MatteQualityError("named region kind is invalid")
    mask = np.asarray(value.mask)
    if mask.ndim != 2 or mask.shape != shape:
        raise MatteQualityError(f"named region {name!r} shape is invalid")
    normalized = np.ascontiguousarray(mask.astype(bool).astype(np.uint8))
    if not bool(np.any(normalized)):
        raise MatteQualityError(f"named region {name!r} cannot be empty")
    return name, QualityNamedRegion(kind=value.kind, mask=normalized)


def write_quality_annotations(
    root: Path | str,
    source_bundle: Path | str | MatteReplayBundle,
    frames: Sequence[QualityFrameAnnotations],
    *,
    segments: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
    gates: Sequence[Mapping[str, object]] = (),
) -> dict[str, Any]:
    """Create a private, digest-bound annotation bundle.

    This helper supports generated CI evidence and consented/licensed local
    qualification.  Callers retain responsibility for recording a meaningful
    provenance/consent reference without embedding private footage or identity.
    """

    bundle = (
        source_bundle
        if isinstance(source_bundle, MatteReplayBundle)
        else MatteReplayBundle(source_bundle)
    )
    if len(frames) != len(bundle.frames):
        raise MatteQualityError("annotation frame count must match the replay bundle")
    if not isinstance(provenance.get("kind"), str) or not isinstance(
        provenance.get("license"), str
    ):
        raise MatteQualityError("annotation provenance requires kind and license")

    output = Path(root)
    _private_directory(output, create=True)
    arrays_dir = output / "arrays"
    _private_directory(arrays_dir, create=True)
    entries: list[dict[str, Any]] = []
    for sequence, (bundle_frame, frame_annotations) in enumerate(
        zip(bundle.frames, frames)
    ):
        final_descriptor = cast(
            dict[str, Any],
            cast(dict[str, Any], bundle_frame["artifacts"])["final_composite"],
        )
        if "alias_of" in final_descriptor:
            raise MatteQualityError("final composite cannot be an artifact alias")
        shape_value = final_descriptor.get("shape")
        if (
            not isinstance(shape_value, list)
            or len(shape_value) != 3
            or any(type(item) is not int for item in shape_value)
        ):
            raise MatteQualityError("source frame shape is unavailable")
        shape = (int(shape_value[0]), int(shape_value[1]))
        registration = frame_annotations.registration_from_previous
        if len(registration) != 6 or any(
            not math.isfinite(float(value)) for value in registration
        ):
            raise MatteQualityError("registration affine must contain six scalars")
        artifacts: dict[str, dict[str, Any]] = {}
        for kind in _ANNOTATION_KINDS:
            array = _validated_annotation_array(
                getattr(frame_annotations, kind),
                kind=kind,
                shape=shape,
            )
            if array is None:
                continue
            filename = f"{sequence:08d}-{kind}.npy"
            payload = _npy_bytes(array)
            _private_write(arrays_dir / filename, payload)
            artifacts[kind] = _array_descriptor(
                f"arrays/{filename}",
                payload,
                array,
            )
        if not isinstance(frame_annotations.regions, Mapping):
            raise MatteQualityError("named regions must be a mapping")
        regions: list[dict[str, Any]] = []
        for region_name, region_value in sorted(
            frame_annotations.regions.items(),
            key=lambda item: str(item[0]),
        ):
            name, region = _validated_region(
                region_name,
                region_value,
                shape=shape,
            )
            filename = f"{sequence:08d}-region-{name}.npy"
            payload = _npy_bytes(region.mask)
            _private_write(arrays_dir / filename, payload)
            regions.append(
                {
                    "name": name,
                    "kind": region.kind,
                    "artifact": _array_descriptor(
                        f"arrays/{filename}",
                        payload,
                        region.mask,
                    ),
                }
            )
        entry = {
            "sequence": sequence,
            "segment": frame_annotations.segment,
            "registration_from_previous": [
                _round(float(value)) for value in registration
            ],
            "artifacts": artifacts,
        }
        if regions:
            entry["regions"] = regions
        entries.append(entry)
    manifest: dict[str, Any] = {
        "schema": ANNOTATION_SCHEMA,
        "version": ANNOTATION_VERSION,
        "source_bundle": {
            "schema": "custback.matte-replay",
            "version": 1,
            "manifest_sha256": _bundle_manifest_digest(bundle),
        },
        "provenance": dict(provenance),
        "segments": [dict(segment) for segment in segments],
        "gates": [dict(gate) for gate in gates],
        "frame_count": len(entries),
        "frames": entries,
    }
    _atomic_private_write(output / "annotations.json", _json_bytes(manifest))
    return manifest


class MatteQualityAnnotations:
    """Validated read-only annotations bound to one replay manifest digest."""

    def __init__(
        self,
        root: Path | str,
        bundle: MatteReplayBundle,
    ):
        self.root = Path(root)
        _private_directory(self.root, create=False)
        payload = _read_private_file(
            self.root / "annotations.json",
            max_bytes=MAX_MANIFEST_BYTES,
        )
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MatteQualityError("annotation manifest is malformed") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != ANNOTATION_SCHEMA
            or manifest.get("version") != ANNOTATION_VERSION
        ):
            raise MatteQualityError("unsupported annotation bundle version")
        source = manifest.get("source_bundle")
        if (
            not isinstance(source, dict)
            or source.get("schema") != "custback.matte-replay"
            or source.get("version") != 1
            or source.get("manifest_sha256") != _bundle_manifest_digest(bundle)
        ):
            raise MatteQualityError("annotations do not match the replay manifest")
        provenance = manifest.get("provenance")
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("kind"), str)
            or not isinstance(provenance.get("license"), str)
        ):
            raise MatteQualityError("annotation provenance is invalid")
        segments = self._validate_segments(manifest.get("segments"), len(bundle.frames))
        gates = manifest.get("gates")
        if not isinstance(gates, list) or not all(
            isinstance(gate, dict) for gate in gates
        ):
            raise MatteQualityError("annotation gates are invalid")
        frames = manifest.get("frames")
        if (
            not isinstance(frames, list)
            or manifest.get("frame_count") != len(frames)
            or len(frames) != len(bundle.frames)
        ):
            raise MatteQualityError("annotation frame count is invalid")
        validated: list[dict[str, Any]] = []
        segment_by_id = {str(segment["id"]): segment for segment in segments}
        for sequence, frame in enumerate(frames):
            validated_frame = self._validate_frame(
                frame,
                sequence,
                segment_ids=set(segment_by_id),
            )
            segment = segment_by_id[str(validated_frame["segment"])]
            if not (
                int(segment["start_sequence"])
                <= sequence
                <= int(segment["end_sequence"])
            ):
                raise MatteQualityError(
                    "annotation frame falls outside its segment bounds"
                )
            validated.append(validated_frame)
        self.manifest = manifest
        self.manifest_sha256 = _sha256(payload)
        self.frames = tuple(validated)
        self.segments = tuple(segments)
        self.gates = tuple(cast(dict[str, Any], gate) for gate in gates)
        self.provenance = provenance

    @staticmethod
    def _validate_segments(
        value: object,
        frame_count: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise MatteQualityError("annotation segments are missing")
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for segment in value:
            if not isinstance(segment, dict):
                raise MatteQualityError("annotation segment is invalid")
            segment_id = segment.get("id")
            kind = segment.get("kind")
            start = segment.get("start_sequence")
            end = segment.get("end_sequence")
            if (
                not isinstance(segment_id, str)
                or not segment_id
                or segment_id in seen
                or kind not in ("stationary", "moving", "fast_motion", "occlusion")
                or type(start) is not int
                or type(end) is not int
                or start < 0
                or end < start
                or end >= frame_count
            ):
                raise MatteQualityError("annotation segment bounds are invalid")
            seen.add(segment_id)
            result.append(dict(segment))
        return result

    @staticmethod
    def _validate_frame(
        value: object,
        sequence: int,
        *,
        segment_ids: set[str],
    ) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or value.get("sequence") != sequence
            or value.get("segment") not in segment_ids
        ):
            raise MatteQualityError("annotation frame identity is invalid")
        affine = value.get("registration_from_previous")
        if (
            not isinstance(affine, list)
            or len(affine) != 6
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in affine
            )
        ):
            raise MatteQualityError("annotation registration is invalid")
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, dict):
            raise MatteQualityError("annotation artifacts are invalid")
        paths: set[PurePosixPath] = set()
        for kind, descriptor in artifacts.items():
            if kind not in _ANNOTATION_KINDS or not isinstance(descriptor, dict):
                raise MatteQualityError("annotation artifact is unknown")
            relative = _safe_annotation_path(descriptor.get("path"))
            if relative in paths:
                raise MatteQualityError("annotation artifact path is duplicated")
            paths.add(relative)
            if (
                type(descriptor.get("bytes")) is not int
                or int(descriptor["bytes"]) <= 0
                or int(descriptor["bytes"]) > MAX_ARTIFACT_BYTES
                or not isinstance(descriptor.get("sha256"), str)
                or len(str(descriptor["sha256"])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(descriptor["sha256"])
                )
                or not isinstance(descriptor.get("dtype"), str)
                or not isinstance(descriptor.get("shape"), list)
            ):
                raise MatteQualityError("annotation artifact metadata is invalid")
        regions = value.get("regions", [])
        if not isinstance(regions, list):
            raise MatteQualityError("annotation named regions are invalid")
        names: set[str] = set()
        for region in regions:
            if not isinstance(region, dict):
                raise MatteQualityError("annotation named region is invalid")
            name = region.get("name")
            kind = region.get("kind")
            descriptor = region.get("artifact")
            if (
                not isinstance(name, str)
                or _REGION_NAME.fullmatch(name) is None
                or name in names
                or kind not in _REGION_KINDS
                or not isinstance(descriptor, dict)
            ):
                raise MatteQualityError("annotation named region metadata is invalid")
            names.add(name)
            relative = _safe_annotation_path(descriptor.get("path"))
            if relative in paths:
                raise MatteQualityError("annotation artifact path is duplicated")
            paths.add(relative)
            if (
                type(descriptor.get("bytes")) is not int
                or int(descriptor["bytes"]) <= 0
                or int(descriptor["bytes"]) > MAX_ARTIFACT_BYTES
                or not isinstance(descriptor.get("sha256"), str)
                or len(str(descriptor["sha256"])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(descriptor["sha256"])
                )
                or descriptor.get("dtype") != np.dtype(np.uint8).str
                or not isinstance(descriptor.get("shape"), list)
            ):
                raise MatteQualityError(
                    "annotation named region artifact metadata is invalid"
                )
        return value

    def load_array(
        self,
        sequence: int,
        kind: AnnotationKind,
        *,
        shape: tuple[int, int],
    ) -> np.ndarray | None:
        frame = self.frames[sequence]
        artifacts = cast(dict[str, Any], frame["artifacts"])
        descriptor = artifacts.get(kind)
        if not isinstance(descriptor, dict):
            return None
        relative = _safe_annotation_path(descriptor["path"])
        path = self.root.joinpath(*relative.parts)
        payload = _read_private_file(path, max_bytes=int(descriptor["bytes"]))
        if (
            len(payload) != descriptor["bytes"]
            or _sha256(payload) != descriptor["sha256"]
        ):
            raise MatteQualityError("annotation artifact integrity check failed")
        try:
            array = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise MatteQualityError(
                "annotation artifact is not a safe NPY array"
            ) from exc
        if (
            array.dtype.str != descriptor["dtype"]
            or list(array.shape) != descriptor["shape"]
        ):
            raise MatteQualityError("annotation artifact metadata does not match")
        return _validated_annotation_array(array, kind=kind, shape=shape)

    def load_regions(
        self,
        sequence: int,
        *,
        shape: tuple[int, int],
    ) -> dict[str, QualityNamedRegion]:
        """Load and integrity-check the optional named regions for one frame."""

        raw_regions = self.frames[sequence].get("regions", [])
        if not isinstance(raw_regions, list):
            raise MatteQualityError("annotation named regions are invalid")
        result: dict[str, QualityNamedRegion] = {}
        for raw_region in raw_regions:
            if not isinstance(raw_region, dict):
                raise MatteQualityError("annotation named region is invalid")
            name = str(raw_region["name"])
            kind = cast(RegionKind, raw_region["kind"])
            descriptor = cast(dict[str, Any], raw_region["artifact"])
            relative = _safe_annotation_path(descriptor["path"])
            payload = _read_private_file(
                self.root.joinpath(*relative.parts),
                max_bytes=int(descriptor["bytes"]),
            )
            if (
                len(payload) != descriptor["bytes"]
                or _sha256(payload) != descriptor["sha256"]
            ):
                raise MatteQualityError(
                    "annotation named region integrity check failed"
                )
            try:
                array = np.load(io.BytesIO(payload), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise MatteQualityError(
                    "annotation named region is not a safe NPY array"
                ) from exc
            if (
                array.dtype.str != descriptor["dtype"]
                or list(array.shape) != descriptor["shape"]
            ):
                raise MatteQualityError(
                    "annotation named region metadata does not match"
                )
            _, value = _validated_region(
                name,
                QualityNamedRegion(kind=kind, mask=array),
                shape=shape,
            )
            result[name] = value
        return result


def _fps(timestamps_ns: Sequence[int]) -> float | None:
    if len(timestamps_ns) < 2:
        return None
    span = timestamps_ns[-1] - timestamps_ns[0]
    if span <= 0:
        return None
    return _round((len(timestamps_ns) - 1) * 1_000_000_000.0 / span)


def _rate(count: int, timestamps_ns: Sequence[int]) -> float | None:
    if len(timestamps_ns) < 2:
        return None
    span = timestamps_ns[-1] - timestamps_ns[0]
    if span <= 0:
        return None
    return _round(count * 1_000_000_000.0 / span)


def _affine(value: Sequence[object]) -> np.ndarray:
    return np.asarray(cast(Sequence[float], value), dtype=np.float32).reshape(2, 3)


def _registration_from_source(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY).astype(np.float32)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _response = cv2.phaseCorrelate(previous_gray, current_gray)
    if not math.isfinite(dx) or not math.isfinite(dy):
        dx = dy = 0.0
    return np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)


def _warp(
    array: np.ndarray,
    affine: np.ndarray,
    shape: tuple[int, int],
    *,
    nearest: bool = False,
) -> np.ndarray:
    return cv2.warpAffine(
        array,
        affine,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _signed_distance(binary: np.ndarray) -> np.ndarray:
    foreground = binary.astype(np.uint8)
    background = (1 - foreground).astype(np.uint8)
    inside = cv2.distanceTransform(foreground, cv2.DIST_L2, cv2.DIST_MASK_5)
    outside = cv2.distanceTransform(background, cv2.DIST_L2, cv2.DIST_MASK_5)
    # OpenCV distances are measured from pixel centers.  Offset both sides by
    # half a pixel so the zero level lies on the foreground/background cell
    # boundary; otherwise a two-pixel contour translation reads as three.
    return np.where(foreground.astype(bool), inside - 0.5, -outside + 0.5)


def _contour_band(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(
        binary.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        kernel,
    ).astype(bool)


def _contour_displacement(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[float | None, float | None]:
    previous_binary = previous >= 0.5
    current_binary = current >= 0.5
    band = _contour_band(previous_binary) | _contour_band(current_binary)
    if not bool(np.any(band)):
        return None, None
    delta = np.abs(
        _signed_distance(current_binary) - _signed_distance(previous_binary)
    )[band]
    return (
        _round(float(np.percentile(delta, 50))),
        _round(float(np.percentile(delta, 95))),
    )


def _perimeter(binary: np.ndarray) -> float:
    contours, _hierarchy = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def _component_count(binary: np.ndarray) -> int:
    if not bool(np.any(binary)):
        return 0
    count, _labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return max(0, int(count) - 1)


def _mean_abs(first: np.ndarray, second: np.ndarray) -> float:
    return _round(
        float(
            np.mean(
                np.abs(first.astype(np.float32) - second.astype(np.float32)),
                dtype=np.float64,
            )
        )
    )


def _region_values(alpha: np.ndarray, region: np.ndarray) -> np.ndarray:
    selected = alpha[region.astype(bool)]
    return selected.astype(np.float64, copy=False)


def _region_metrics(
    *,
    raw: np.ndarray,
    alpha: np.ndarray,
    backdrop: np.ndarray,
    composite: np.ndarray,
    clean_foreground: np.ndarray | None,
    opaque: np.ndarray | None,
    background: np.ndarray | None,
    ground_truth_alpha: np.ndarray | None,
    ground_truth_foreground: np.ndarray | None,
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "opaque_core_alpha_p05": None,
        "opaque_core_alpha_p50": None,
        "opaque_core_mean_deficit": None,
        "opaque_core_fraction_below_0_95": None,
        "opaque_core_fraction_below_0_90": None,
        "background_alpha_mass": None,
        "background_alpha_mean": None,
        "exterior_halo_area_ratio": None,
        "exterior_halo_width_p95_px": None,
        "foreground_hole_components": None,
        "unexpected_foreground_components": None,
        "opaque_backdrop_leakage_coefficient": None,
        "opaque_backdrop_correlation": None,
        "opaque_composite_source_mae": None,
        "clean_foreground_rgb_error": None,
    }
    if opaque is not None and bool(np.any(opaque)):
        values = _region_values(alpha, opaque)
        metrics.update(
            {
                "opaque_core_alpha_p05": _round(float(np.percentile(values, 5))),
                "opaque_core_alpha_p50": _round(float(np.percentile(values, 50))),
                "opaque_core_mean_deficit": _round(float(np.mean(1.0 - values))),
                "opaque_core_fraction_below_0_95": _round(
                    float(np.mean(values < 0.95))
                ),
                "opaque_core_fraction_below_0_90": _round(
                    float(np.mean(values < 0.90))
                ),
                "foreground_hole_components": _component_count(
                    opaque.astype(bool) & (alpha < 0.5)
                ),
            }
        )
        selected = opaque.astype(bool)
        source_pixels = raw[selected].astype(np.float64)
        backdrop_pixels = backdrop[selected].astype(np.float64)
        composite_pixels = composite[selected].astype(np.float64)
        direction = backdrop_pixels - source_pixels
        change = composite_pixels - source_pixels
        denominator = float(np.sum(direction * direction, dtype=np.float64))
        leakage = (
            float(np.sum(change * direction, dtype=np.float64)) / denominator
            if denominator > 0.0
            else 0.0
        )
        flat_backdrop = backdrop_pixels.reshape(-1)
        flat_composite = composite_pixels.reshape(-1)
        backdrop_centered = flat_backdrop - float(np.mean(flat_backdrop))
        composite_centered = flat_composite - float(np.mean(flat_composite))
        correlation_denominator = math.sqrt(
            float(np.sum(backdrop_centered * backdrop_centered))
            * float(np.sum(composite_centered * composite_centered))
        )
        correlation = (
            float(np.sum(backdrop_centered * composite_centered))
            / correlation_denominator
            if correlation_denominator > 0.0
            else 0.0
        )
        metrics.update(
            {
                "opaque_backdrop_leakage_coefficient": _round(leakage),
                "opaque_backdrop_correlation": _round(correlation),
                "opaque_composite_source_mae": _round(
                    float(np.mean(np.abs(composite_pixels - source_pixels))) / 255.0
                ),
            }
        )
    if background is not None and bool(np.any(background)):
        values = _region_values(alpha, background)
        false_foreground = background.astype(bool) & (alpha > 0.05)
        metrics.update(
            {
                "background_alpha_mass": _round(float(np.sum(values))),
                "background_alpha_mean": _round(float(np.mean(values))),
                "exterior_halo_area_ratio": _round(float(np.mean(values > 0.05))),
                "unexpected_foreground_components": _component_count(
                    background.astype(bool) & (alpha >= 0.5)
                ),
            }
        )
        if ground_truth_alpha is not None and bool(np.any(false_foreground)):
            exterior = (ground_truth_alpha <= 0.05).astype(np.uint8)
            exterior_distance = cv2.distanceTransform(
                exterior,
                cv2.DIST_L2,
                cv2.DIST_MASK_5,
            )
            metrics["exterior_halo_width_p95_px"] = _round(
                float(np.percentile(exterior_distance[false_foreground], 95))
            )
    if (
        clean_foreground is not None
        and ground_truth_foreground is not None
        and ground_truth_alpha is not None
    ):
        replacement = (ground_truth_alpha > 0.05) & (ground_truth_alpha < 0.95)
        if bool(np.any(replacement)):
            metrics["clean_foreground_rgb_error"] = _round(
                float(
                    np.mean(
                        np.abs(
                            clean_foreground[replacement].astype(np.float32)
                            - ground_truth_foreground[replacement].astype(np.float32)
                        ),
                        dtype=np.float64,
                    )
                )
                / 255.0
            )
    return metrics


def _ground_truth_metrics(
    alpha: np.ndarray,
    ground_truth: np.ndarray | None,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "ground_truth_alpha_sad": None,
        "ground_truth_alpha_mae": None,
        "ground_truth_alpha_mse": None,
        "ground_truth_gradient_mae": None,
    }
    if ground_truth is None:
        return result
    delta = alpha.astype(np.float64) - ground_truth.astype(np.float64)
    gradient_alpha_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    gradient_alpha_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    gradient_gt_x = cv2.Sobel(ground_truth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_gt_y = cv2.Sobel(ground_truth, cv2.CV_32F, 0, 1, ksize=3)
    gradient_delta = np.hypot(
        gradient_alpha_x - gradient_gt_x,
        gradient_alpha_y - gradient_gt_y,
    )
    result.update(
        {
            "ground_truth_alpha_sad": _round(float(np.sum(np.abs(delta)))),
            "ground_truth_alpha_mae": _round(float(np.mean(np.abs(delta)))),
            "ground_truth_alpha_mse": _round(float(np.mean(delta * delta))),
            "ground_truth_gradient_mae": _round(
                float(np.mean(gradient_delta, dtype=np.float64))
            ),
        }
    )
    return result


def _artifact_bytes(frame: Mapping[str, object]) -> int:
    artifacts = cast(dict[str, Any], frame["artifacts"])
    return sum(
        int(descriptor["bytes"])
        for descriptor in artifacts.values()
        if isinstance(descriptor, dict) and "alias_of" not in descriptor
    )


def _metric_path(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _finite_number(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def evaluate_gates(
    report: Mapping[str, object],
    gates: Sequence[Mapping[str, object]],
    *,
    baseline: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate data-defined absolute or baseline-relative release gates."""

    results: list[dict[str, Any]] = []
    for gate in gates:
        gate_id = gate.get("id")
        path = gate.get("metric")
        operation = gate.get("op")
        if (
            not isinstance(gate_id, str)
            or not gate_id
            or not isinstance(path, str)
            or operation not in ("<=", ">=")
        ):
            raise MatteQualityError("quality gate definition is invalid")
        actual = _metric_path(report, path)
        threshold = gate.get("value")
        baseline_value: object = None
        multiplier = gate.get("baseline_multiplier")
        if multiplier is not None:
            multiplier_number = _finite_number(multiplier)
            if multiplier_number is None:
                raise MatteQualityError("quality gate baseline multiplier is invalid")
            if baseline is not None:
                baseline_value = _metric_path(baseline, path)
                baseline_number = _finite_number(baseline_value)
                if baseline_number is not None:
                    threshold = baseline_number * multiplier_number
                else:
                    threshold = None
            else:
                threshold = None
        actual_number = _finite_number(actual)
        threshold_number = _finite_number(threshold)
        if actual_number is not None and threshold_number is not None:
            passed = (
                actual_number <= threshold_number
                if operation == "<="
                else actual_number >= threshold_number
            )
            status = "pass" if passed else "fail"
        else:
            status = "not_evaluated"
        results.append(
            {
                "id": gate_id,
                "metric": path,
                "op": operation,
                "actual": (
                    _round(actual_number) if actual_number is not None else actual
                ),
                "threshold": (
                    _round(threshold_number) if threshold_number is not None else None
                ),
                "baseline": (
                    _round(cast(float, _finite_number(baseline_value)))
                    if _finite_number(baseline_value) is not None
                    else None
                ),
                "status": status,
                "notes": str(gate.get("notes", "")),
            }
        )
    return results


def _environment(metadata: EvaluationMetadata) -> dict[str, object]:
    return {
        "hardware": {
            "label": metadata.hardware_label,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "backend": metadata.backend,
        "device": metadata.device,
        "effective_detail": metadata.effective_detail,
        "resampling": metadata.resampling,
        "configuration_label": metadata.configuration_label,
        "notes": metadata.notes,
    }


def _cadence_metrics(bundle: MatteReplayBundle) -> dict[str, object]:
    input_timestamps = [int(frame["capture_monotonic_ns"]) for frame in bundle.frames]
    capture_sequences = [int(frame["capture_sequence"]) for frame in bundle.frames]
    missing_inputs = sum(
        max(0, current - previous - 1)
        for previous, current in zip(capture_sequences, capture_sequences[1:])
    )
    events = bundle.output_events
    if not events:
        return {
            "unique_input_count": len(bundle.frames),
            "unique_input_fps": _fps(input_timestamps),
            "capture_sequence_gap_count": sum(
                current - previous > 1
                for previous, current in zip(capture_sequences, capture_sequences[1:])
            ),
            "capture_missing_input_count": missing_inputs,
            "output_timeline_available": False,
            "output_timeline_complete": False,
            "output_send_count": 0,
            "output_send_fps": None,
            "output_repeat_ratio": None,
            "exact_final_output_repeat_count": None,
            "base_composite_update_count": None,
            "base_composite_reuse_count": None,
            "base_composite_update_fps": None,
            "base_composite_reuse_fps": None,
        }
    output_timestamps = [int(event["sent_monotonic_ns"]) for event in events]
    final_digests = {
        int(frame["sequence"]): cast(dict[str, Any], frame["artifacts"])[
            "final_composite"
        ]["sha256"]
        for frame in bundle.frames
    }
    exact_repeat_count = 0
    previous_digest: str | None = None
    for event in events:
        digest = str(final_digests[int(event["source_bundle_sequence"])])
        if previous_digest is not None and digest == previous_digest:
            exact_repeat_count += 1
        previous_digest = digest
    update_count = sum(bool(event["base_updated"]) for event in events)
    reuse_count = len(events) - update_count
    transitions = max(1, len(events) - 1)
    timeline = cast(dict[str, Any], bundle.manifest.get("output_timeline", {}))
    return {
        "unique_input_count": len(bundle.frames),
        "unique_input_fps": _fps(input_timestamps),
        "capture_sequence_gap_count": sum(
            current - previous > 1
            for previous, current in zip(capture_sequences, capture_sequences[1:])
        ),
        "capture_missing_input_count": missing_inputs,
        "output_timeline_available": True,
        "output_timeline_complete": bool(timeline.get("complete")),
        "output_send_count": len(events),
        "output_send_fps": _fps(output_timestamps),
        "output_repeat_ratio": _round(exact_repeat_count / transitions),
        "exact_final_output_repeat_count": exact_repeat_count,
        "base_composite_update_count": update_count,
        "base_composite_reuse_count": reuse_count,
        "base_composite_update_fps": _rate(
            sum(bool(event["base_updated"]) for event in events[1:]),
            output_timestamps,
        ),
        "base_composite_reuse_fps": _rate(
            sum(not bool(event["base_updated"]) for event in events[1:]),
            output_timestamps,
        ),
    }


def evaluate_bundle(
    bundle_root: Path | str,
    *,
    annotations_root: Path | str | None = None,
    metadata: EvaluationMetadata = EvaluationMetadata(),
    baseline: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Evaluate a complete replay bundle without invoking a live model."""

    bundle = MatteReplayBundle(bundle_root)
    if not bundle.frames:
        raise MatteQualityError("quality evaluation requires at least one frame")
    if not bool(bundle.manifest.get("matte_metrics_authoritative")):
        raise MatteQualityError(
            "quality evaluation requires a full metric-authoritative bundle"
        )
    annotations = (
        MatteQualityAnnotations(annotations_root, bundle)
        if annotations_root is not None
        else None
    )

    per_frame: list[dict[str, Any]] = []
    all_series = _MetricSeries()
    segment_series: dict[str, _MetricSeries] = {}
    kind_series: dict[str, _MetricSeries] = {}
    segment_by_id = (
        {str(segment["id"]): segment for segment in annotations.segments}
        if annotations is not None
        else {}
    )
    stationary_reference_area: dict[str, float] = {}
    previous_raw: np.ndarray | None = None
    previous_raw_mask: np.ndarray | None = None
    previous_refined: np.ndarray | None = None
    previous_composite: np.ndarray | None = None
    previous_ground_truth: np.ndarray | None = None
    trail_run = 0
    max_trail_run = 0
    timing_series: dict[str, list[float]] = {}
    allocation_series: list[float] = []
    memory_series: list[float] = []
    artifact_series: list[float] = []
    evaluator_array_series: list[float] = []

    for sequence, frame in enumerate(bundle.frames):
        raw = bundle.load_array(frame, "raw_frame")
        raw_mask = bundle.load_array(frame, "raw_mask").astype(np.float32, copy=False)
        refined = bundle.load_array(frame, "refined_mask").astype(
            np.float32,
            copy=False,
        )
        backdrop = bundle.load_array(frame, "backdrop_frame")
        composite = bundle.load_array(frame, "final_composite")
        artifacts = cast(dict[str, Any], frame["artifacts"])
        clean_foreground = (
            bundle.load_array(frame, "clean_foreground")
            if "clean_foreground" in artifacts
            else None
        )
        shape = refined.shape
        if (
            raw.shape != (*shape, 3)
            or raw_mask.shape != shape
            or backdrop.shape != raw.shape
            or composite.shape != raw.shape
        ):
            raise MatteQualityError("metric-authoritative tracks do not align")

        annotation_frame = (
            annotations.frames[sequence] if annotations is not None else None
        )
        segment_id = (
            str(annotation_frame["segment"])
            if annotation_frame is not None
            else "unannotated"
        )
        segment_kind = (
            str(segment_by_id[segment_id]["kind"])
            if segment_id in segment_by_id
            else "unannotated"
        )
        if annotations is not None:
            opaque = annotations.load_array(sequence, "opaque_core", shape=shape)
            background = annotations.load_array(sequence, "background", shape=shape)
            ground_truth = annotations.load_array(
                sequence,
                "ground_truth_alpha",
                shape=shape,
            )
            ground_truth_foreground = annotations.load_array(
                sequence,
                "ground_truth_foreground",
                shape=shape,
            )
            assert annotation_frame is not None
            registration = _affine(annotation_frame["registration_from_previous"])
            registration_method = "annotation-affine"
        else:
            opaque = background = ground_truth = ground_truth_foreground = None
            registration = (
                _registration_from_source(previous_raw, raw)
                if previous_raw is not None
                else np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                )
            )
            registration_method = "source-phase-correlation"

        uncertain = (refined > 0.05) & (refined < 0.95)
        perimeter = _perimeter(refined >= 0.5)
        frame_metrics: dict[str, float | int | None] = {
            "raw_alpha_temporal_abs_diff": None,
            "refined_alpha_temporal_abs_diff": None,
            "compensated_alpha_temporal_abs_diff": None,
            "contour_displacement_p50_px": None,
            "contour_displacement_p95_px": None,
            "stationary_subject_area_drift": None,
            "soft_edge_width_px": (
                _round(float(np.count_nonzero(uncertain)) / perimeter)
                if perimeter > 0.0
                else 0.0
            ),
            "uncertain_pixel_fraction": _round(
                float(np.mean(uncertain, dtype=np.float64))
            ),
            "motion_trail_area_ratio": None,
            "edge_band_rgb_variation": None,
            "recorded_artifact_bytes": _artifact_bytes(frame),
        }
        current_area = float(np.sum(refined, dtype=np.float64))
        if segment_kind == "stationary":
            reference_area = stationary_reference_area.setdefault(
                segment_id,
                current_area,
            )
            frame_metrics["stationary_subject_area_drift"] = (
                _round(abs(current_area - reference_area) / reference_area)
                if reference_area > 0.0
                else 0.0
            )

        if previous_refined is not None and previous_raw_mask is not None:
            warped_refined = _warp(previous_refined, registration, shape)
            frame_metrics.update(
                {
                    "raw_alpha_temporal_abs_diff": _mean_abs(
                        raw_mask,
                        previous_raw_mask,
                    ),
                    "refined_alpha_temporal_abs_diff": _mean_abs(
                        refined,
                        previous_refined,
                    ),
                    "compensated_alpha_temporal_abs_diff": _mean_abs(
                        refined,
                        warped_refined,
                    ),
                }
            )
            contour_p50, contour_p95 = _contour_displacement(
                warped_refined,
                refined,
            )
            frame_metrics["contour_displacement_p50_px"] = contour_p50
            frame_metrics["contour_displacement_p95_px"] = contour_p95
            if previous_composite is not None:
                warped_composite = _warp(previous_composite, registration, shape)
                held_alpha = np.abs(refined - warped_refined) <= 0.01
                edge_band = held_alpha & uncertain
                if bool(np.any(edge_band)):
                    frame_metrics["edge_band_rgb_variation"] = _round(
                        float(
                            np.mean(
                                np.abs(
                                    composite[edge_band].astype(np.float32)
                                    - warped_composite[edge_band].astype(np.float32)
                                ),
                                dtype=np.float64,
                            )
                        )
                        / 255.0
                    )
        if ground_truth is not None and previous_ground_truth is not None:
            prior_only = (previous_ground_truth >= 0.5) & (ground_truth < 0.5)
            trailing = prior_only & (refined >= 0.5)
            current_ground_truth_area = int(np.count_nonzero(ground_truth >= 0.5))
            trail_ratio = (
                float(np.count_nonzero(trailing)) / current_ground_truth_area
                if current_ground_truth_area > 0
                else 0.0
            )
            frame_metrics["motion_trail_area_ratio"] = _round(trail_ratio)
            if trail_ratio > 0.01:
                trail_run += 1
                max_trail_run = max(max_trail_run, trail_run)
            else:
                trail_run = 0

        frame_metrics.update(
            _region_metrics(
                raw=raw,
                alpha=refined,
                backdrop=backdrop,
                composite=composite,
                clean_foreground=clean_foreground,
                opaque=opaque,
                background=background,
                ground_truth_alpha=ground_truth,
                ground_truth_foreground=ground_truth_foreground,
            )
        )
        frame_metrics.update(_ground_truth_metrics(refined, ground_truth))

        for name, value in cast(dict[str, Any], frame.get("timings_ms", {})).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timing_series.setdefault(str(name), []).append(float(value))
        for name, value in cast(
            dict[str, Any],
            frame.get("compositor_substages_ms", {}),
        ).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timing_series.setdefault(f"compositor.{name}", []).append(float(value))
        resources = frame.get("resource_samples")
        if isinstance(resources, dict):
            allocation = resources.get("allocation_bytes")
            memory = resources.get("memory_bytes")
            if isinstance(allocation, (int, float)) and not isinstance(
                allocation, bool
            ):
                allocation_series.append(float(allocation))
            if isinstance(memory, (int, float)) and not isinstance(memory, bool):
                memory_series.append(float(memory))
        recorded_bytes = frame_metrics["recorded_artifact_bytes"]
        assert isinstance(recorded_bytes, int)
        artifact_series.append(float(recorded_bytes))
        evaluator_array_series.append(
            float(
                sum(
                    array.nbytes
                    for array in (
                        raw,
                        raw_mask,
                        refined,
                        backdrop,
                        composite,
                        clean_foreground,
                        opaque,
                        background,
                        ground_truth,
                        ground_truth_foreground,
                    )
                    if array is not None
                )
            )
        )

        all_series.add(frame_metrics)
        segment_series.setdefault(segment_id, _MetricSeries()).add(frame_metrics)
        kind_series.setdefault(segment_kind, _MetricSeries()).add(frame_metrics)
        per_frame.append(
            {
                "sequence": sequence,
                "capture_sequence": frame["capture_sequence"],
                "capture_monotonic_ns": frame["capture_monotonic_ns"],
                "segment": segment_id,
                "segment_kind": segment_kind,
                "registration": {
                    "method": registration_method,
                    "previous_to_current_affine": [
                        _round(float(value)) for value in registration.reshape(-1)
                    ],
                },
                "metrics": frame_metrics,
            }
        )
        previous_raw = raw
        previous_raw_mask = raw_mask
        previous_refined = refined
        previous_composite = composite
        previous_ground_truth = ground_truth

    cadence = _cadence_metrics(bundle)
    timing_summaries = {
        name: _summary(values) for name, values in sorted(timing_series.items())
    }
    aggregate = {
        "cadence": cadence,
        "metrics": all_series.summaries(),
        "segments": {
            name: {
                "kind": str(segment_by_id.get(name, {}).get("kind", "unannotated")),
                "metrics": series.summaries(),
            }
            for name, series in sorted(segment_series.items())
        },
        "segment_kinds": {
            name: {"metrics": series.summaries()}
            for name, series in sorted(kind_series.items())
        },
        "motion": {
            "maximum_dominant_previous_contour_intervals": max_trail_run,
        },
        "performance": {
            "timings_ms": timing_summaries,
            "runtime_allocation_bytes": {
                "available": bool(allocation_series),
                **_summary(allocation_series),
            },
            "runtime_memory_bytes": {
                "available": bool(memory_series),
                **_summary(memory_series),
            },
            "recorded_artifact_bytes": _summary(artifact_series),
            "evaluator_loaded_array_bytes": _summary(evaluator_array_series),
        },
    }
    post_base_events = [
        {
            "output_sequence": event["sequence"],
            "provenance": event["post_base_final_output_provenance"],
        }
        for event in bundle.output_events
        if event.get("post_base_final_output_provenance") is not None
    ]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "source": {
            "bundle_schema": bundle.manifest["schema"],
            "bundle_version": bundle.manifest["version"],
            "bundle_manifest_sha256": _bundle_manifest_digest(bundle),
            "annotation_manifest_sha256": (
                annotations.manifest_sha256 if annotations is not None else None
            ),
            "annotation_provenance": (
                annotations.provenance if annotations is not None else None
            ),
        },
        "environment": _environment(metadata),
        "configuration": {
            "effective": bundle.frames[0].get("effective_controls", {}),
            "input_cadence": cadence,
        },
        "determinism": {
            "metric_rounding_decimal_places": 8,
            "cross_platform_absolute_tolerance": METRIC_TOLERANCE,
            "timestamp_metrics_use_recorded_monotonic_ns": True,
        },
        "aggregate": aggregate,
        "per_frame": per_frame,
        "extensions": {
            "post_base": {
                "schema": POST_BASE_SCHEMA,
                "version": 1,
                "events": post_base_events,
            }
        },
        "gates": [],
    }
    report["gates"] = evaluate_gates(
        report,
        annotations.gates if annotations is not None else (),
        baseline=baseline,
    )
    deterministic_evidence = {
        "source": report["source"],
        "aggregate": aggregate,
        "per_frame": per_frame,
        "extensions": report["extensions"],
        "gates": report["gates"],
    }
    report["determinism"]["evidence_sha256"] = _sha256(
        _json_bytes(deterministic_evidence)
    )
    return report


def report_markdown(report: Mapping[str, object]) -> str:
    """Render a compact review companion for the canonical JSON report."""

    source = cast(dict[str, Any], report["source"])
    environment = cast(dict[str, Any], report["environment"])
    aggregate = cast(dict[str, Any], report["aggregate"])
    cadence = cast(dict[str, Any], aggregate["cadence"])
    determinism = cast(dict[str, Any], report["determinism"])
    gates = cast(list[dict[str, Any]], report["gates"])
    lines = [
        "# Matte quality report",
        "",
        f"- Evidence SHA-256: `{determinism['evidence_sha256']}`",
        f"- Replay manifest SHA-256: `{source['bundle_manifest_sha256']}`",
        f"- Annotation manifest SHA-256: `{source['annotation_manifest_sha256']}`",
        f"- Hardware: `{environment['hardware']}`",
        f"- Dependencies: `{environment['dependencies']}`",
        f"- Backend/device: `{environment['backend']}` / `{environment['device']}`",
        f"- Detail/resampling: `{environment['effective_detail']}` / "
        f"`{environment['resampling']}`",
        "",
        "## Cadence",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for name in (
        "unique_input_fps",
        "output_send_fps",
        "output_repeat_ratio",
        "base_composite_update_fps",
        "base_composite_reuse_fps",
    ):
        lines.append(f"| `{name}` | `{cadence.get(name)}` |")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Actual | Threshold | Status |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    if gates:
        for gate in gates:
            lines.append(
                f"| `{gate['id']}` | `{gate['actual']}` | "
                f"`{gate['threshold']}` | **{gate['status']}** |"
            )
    else:
        lines.append("| No data-defined gates | — | — | not evaluated |")
    lines.extend(
        [
            "",
            "The JSON report is authoritative. Null runtime allocation/memory "
            "statistics mean the replay did not carry instrumented resource samples; "
            "recorded artifact volume and evaluator working-set estimates remain "
            "available separately.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser(
    *,
    prog: str = "custback matte-evaluate",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Evaluate a private raw/mask/composite replay bundle offline",
    )
    parser.add_argument("bundle", help="complete private replay bundle directory")
    parser.add_argument(
        "--annotations",
        help="private digest-bound annotation bundle directory",
    )
    parser.add_argument("--json", required=True, help="quality report JSON path")
    parser.add_argument("--markdown", help="optional Markdown review report path")
    parser.add_argument(
        "--baseline",
        help="optional prior JSON report for relative data-defined gates",
    )
    parser.add_argument("--hardware-label", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--effective-detail", default="")
    parser.add_argument("--resampling", default="")
    parser.add_argument("--configuration-label", default="")
    parser.add_argument("--notes", default="")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback matte-evaluate",
) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        baseline: Mapping[str, object] | None = None
        if args.baseline:
            baseline_value = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            if (
                not isinstance(baseline_value, dict)
                or baseline_value.get("schema") != REPORT_SCHEMA
                or baseline_value.get("version") != REPORT_VERSION
            ):
                raise MatteQualityError("baseline report schema is invalid")
            baseline = baseline_value
        report = evaluate_bundle(
            args.bundle,
            annotations_root=args.annotations,
            metadata=EvaluationMetadata(
                hardware_label=args.hardware_label,
                backend=args.backend,
                device=args.device,
                effective_detail=args.effective_detail,
                resampling=args.resampling,
                configuration_label=args.configuration_label,
                notes=args.notes,
            ),
            baseline=baseline,
        )
        _atomic_private_write(Path(args.json), _json_bytes(report))
        if args.markdown:
            _atomic_private_write(
                Path(args.markdown),
                report_markdown(report).encode("utf-8"),
            )
    except (
        OSError,
        json.JSONDecodeError,
        MatteDiagnosticsError,
        MatteQualityError,
    ) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    failures = sum(gate["status"] == "fail" for gate in report["gates"])
    print(
        f"evaluated {len(report['per_frame'])} unique frame(s); "
        f"{failures} gate failure(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
