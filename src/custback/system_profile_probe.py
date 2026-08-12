"""Sequential, pixel-free capture screening for catalog system profiles."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, cast

from .capture_diagnostics import (
    CaptureDiagnosticError,
    CaptureFactory,
    _error_code,
    run_capture_only,
)
from .config import AppConfig
from .matte_diagnostics import _atomic_private_write, _json_bytes, _private_directory
from .system_profiles import (
    NONQUALIFIED_EVIDENCE_STATES,
    PROFILE_CATALOG,
    flatten_patch,
)


REPORT_SCHEMA = "custback.system-profile-probe"
REPORT_VERSION = 1


def _profile_config(base: AppConfig, identifier: str) -> AppConfig:
    definition = PROFILE_CATALOG.definition("quality", identifier)
    leaves = flatten_patch(definition.detached_patch())
    values = base.to_dict()
    camera = cast(dict[str, Any], values["camera"])
    output = cast(dict[str, Any], values["output"])
    for path in ("camera.width", "camera.height", "camera.fps"):
        camera[path.split(".")[1]] = leaves[path]
    for path in ("output.width", "output.height", "output.fps"):
        output[path.split(".")[1]] = leaves[path]
    return AppConfig.from_dict(values)


def _suitability(report: Mapping[str, Any], target_fps: int) -> dict[str, Any]:
    requested = cast(Mapping[str, Any], report["requested"])
    negotiated = cast(Mapping[str, Any], report["negotiated"])
    measurement = cast(Mapping[str, Any], report["measurement"])
    timing = cast(Mapping[str, Any], report.get("timing", {}))
    window = cast(Mapping[str, Any], timing.get("measurement_window", {}))
    diagnosis = cast(Mapping[str, Any], report.get("diagnosis", {}))
    pacing = cast(
        Mapping[str, Any], cast(Mapping[str, Any], report["pacing"])["capture"]
    )
    minimum_fps = (
        27.0 if target_fps == 30 else 14.5 if target_fps == 15 else 0.9 * target_fps
    )
    active_fps = pacing.get("active_source_fps")
    wall_fps = pacing.get("wall_completion_fps")
    availability_fps = window.get("availability_fps")

    def _rate(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed >= 0.0 else None

    measured_rates = tuple(
        rate
        for rate in (
            _rate(active_fps),
            _rate(wall_fps),
            _rate(availability_fps),
        )
        if rate is not None
    )
    measured_unique_fps = min(measured_rates) if len(measured_rates) == 3 else None
    exact_geometry = (
        type(requested.get("width")) is int
        and type(requested.get("height")) is int
        and type(negotiated.get("width")) is int
        and type(negotiated.get("height")) is int
        and type(negotiated.get("delivered_width")) is int
        and type(negotiated.get("delivered_height")) is int
        and negotiated["width"] == requested["width"]
        and negotiated["height"] == requested["height"]
        and negotiated["delivered_width"] == requested["width"]
        and negotiated["delivered_height"] == requested["height"]
    )
    cadence_met = bool(
        measured_unique_fps is not None and measured_unique_fps >= minimum_fps
    )
    window_complete = diagnosis.get("measurement_window_complete") is True
    window_sustained = diagnosis.get("measurement_window_sustained") is True
    target_sustained = diagnosis.get("target_sustained") is True
    healthy = (
        measurement.get("capture_error") is None
        and measurement.get("close_error") is None
        and type(measurement.get("read_failures")) is int
        and measurement["read_failures"] == 0
        and type(measurement.get("restarts")) is int
        and measurement["restarts"] == 0
        and type(measurement.get("geometry_transitions")) is int
        and measurement["geometry_transitions"] == 0
        and measurement.get("stalled_at_end") is False
    )
    reasons: list[str] = []
    if not exact_geometry:
        reasons.append("negotiated-geometry-mismatch")
    if not window_complete:
        reasons.append("capture-window-incomplete")
    if not window_sustained:
        reasons.append("capture-window-not-sustained")
    if not cadence_met:
        reasons.append("capture-cadence-shortfall")
    if not target_sustained:
        reasons.append("capture-target-not-sustained")
    if not healthy:
        reasons.append("capture-health-failure")
    return {
        "capture_suitable": bool(
            exact_geometry
            and cadence_met
            and window_complete
            and window_sustained
            and target_sustained
            and healthy
        ),
        "minimum_unique_fps": minimum_fps,
        "measured_unique_fps": measured_unique_fps,
        "active_source_fps": _rate(active_fps),
        "wall_completion_fps": _rate(wall_fps),
        "availability_fps": _rate(availability_fps),
        "exact_geometry": exact_geometry,
        "measurement_window_complete": window_complete,
        "measurement_window_sustained": window_sustained,
        "target_sustained": target_sustained,
        "healthy": healthy,
        "reasons": reasons,
        "full_path_suitability": "not_measured",
        "output_suitability": "not_measured",
    }


def run_system_profile_probe(
    base: AppConfig,
    profile_ids: list[str],
    *,
    accept_experimental: bool,
    condition_id: str = "default",
    hardware_verified: bool = False,
    device_identity_sha256: str = "",
    hardware_identity_sha256: str = "",
    warmup_seconds: float = 2.0,
    measurement_seconds: float = 10.0,
    capture_factory: CaptureFactory | None = None,
) -> dict[str, Any]:
    if not profile_ids or len(profile_ids) != len(set(profile_ids)):
        raise CaptureDiagnosticError(
            "requested profile IDs must be unique and non-empty"
        )
    if accept_experimental is not True:
        raise CaptureDiagnosticError(
            "non-qualified profile probe requires explicit acknowledgment"
        )
    rows: list[dict[str, Any]] = []
    aborted_reason = ""
    for identifier in profile_ids:
        definition = PROFILE_CATALOG.definition("quality", identifier)
        if definition.evidence_state not in NONQUALIFIED_EVIDENCE_STATES:
            raise CaptureDiagnosticError(
                "only non-qualified catalog rows are accepted by this probe"
            )
        cfg = _profile_config(base, identifier)
        kwargs: dict[str, Any] = {}
        if capture_factory is not None:
            kwargs["capture_factory"] = capture_factory
        report = run_capture_only(
            cfg,
            condition_id=condition_id,
            hardware_verified=hardware_verified,
            device_identity_sha256=device_identity_sha256,
            hardware_identity_sha256=hardware_identity_sha256,
            warmup_seconds=warmup_seconds,
            measurement_seconds=measurement_seconds,
            **kwargs,
        )
        measurement = cast(Mapping[str, Any], report["measurement"])
        row = {
            "profile_id": identifier,
            "patch_digest": definition.patch_digest,
            "capture": report,
            "suitability": _suitability(report, cfg.camera.fps),
        }
        rows.append(row)
        if measurement.get("close_error") is not None:
            aborted_reason = "reader-close-failure"
            break
    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "privacy": {
            "contains_pixels": False,
            "contains_frame_hashes": False,
            "contains_wall_clock_timestamps": False,
            "contains_device_path_or_index": False,
            "contains_credentials": False,
        },
        "catalog_version": PROFILE_CATALOG.version,
        "catalog_digest": PROFILE_CATALOG.digest,
        "quality_claim": False,
        "capture_only": True,
        "preferences_mutated": False,
        "full_path_qualification_inferred": False,
        "requested_profiles": list(profile_ids),
        "completed_profiles": len(rows),
        "aborted_reason": aborted_reason,
        "profiles": rows,
    }


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Custback system profile capture probe",
        "",
        "This report is pixel-free and capture-only. It does not qualify the model, compositor, output sink, or consumer path.",
        "",
        "| Profile | Capture suitable | Unique FPS | Required FPS | Reasons |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in cast(list[Mapping[str, Any]], report["profiles"]):
        suitability = cast(Mapping[str, Any], row["suitability"])
        reasons = ", ".join(cast(list[str], suitability["reasons"])) or "none"
        lines.append(
            f"| {row['profile_id']} | {str(suitability['capture_suitable']).lower()} | "
            f"{suitability['measured_unique_fps']} | {suitability['minimum_unique_fps']} | {reasons} |"
        )
    if report.get("aborted_reason"):
        lines.extend(("", f"Probe aborted: `{report['aborted_reason']}`."))
    lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output: Path | str) -> None:
    root = Path(output)
    _private_directory(root, create=True)
    _atomic_private_write(root / "profiles.json", _json_bytes(report))
    _atomic_private_write(root / "profiles.md", report_markdown(report).encode("utf-8"))


def build_parser(
    *, prog: str = "custback system-profile-probe"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--config", help="runtime YAML used for device and capture policy"
    )
    parser.add_argument("--device", help="camera device override (never serialized)")
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(PROFILE_CATALOG.axes["quality"]),
        required=True,
        help="quality profile camera mode to probe; repeat for a matrix",
    )
    parser.add_argument(
        "--accept-experimental",
        action="store_true",
        help=(
            "acknowledge that catalog profiles are experimental or locally "
            "screened, not qualified"
        ),
    )
    parser.add_argument("--pixel-format", choices=("auto", "mjpeg", "backend"))
    parser.add_argument("--mode-mismatch", choices=("warn", "error"))
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--condition-id", default="default")
    parser.add_argument("--hardware-verified", action="store_true")
    parser.add_argument("--device-identity-sha256", default="")
    parser.add_argument("--hardware-identity-sha256", default="")
    parser.add_argument(
        "--output", required=True, help="new owner-only report directory"
    )
    return parser


def main(
    argv: list[str] | None = None, *, prog: str = "custback system-profile-probe"
) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        base = AppConfig.load(args.config)
        values = base.to_dict()
        camera = cast(dict[str, Any], values["camera"])
        if args.device is not None:
            camera["device"] = args.device
        if args.pixel_format is not None:
            camera["pixel_format"] = args.pixel_format
        if args.mode_mismatch is not None:
            camera["mode_mismatch"] = args.mode_mismatch
        base = AppConfig.from_dict(values)
        report = run_system_profile_probe(
            base,
            args.profile,
            accept_experimental=args.accept_experimental,
            condition_id=args.condition_id,
            hardware_verified=args.hardware_verified,
            device_identity_sha256=args.device_identity_sha256,
            hardware_identity_sha256=args.hardware_identity_sha256,
            warmup_seconds=args.warmup_seconds,
            measurement_seconds=args.duration_seconds,
        )
        write_report(report, args.output)
    except KeyboardInterrupt:
        print(f"{prog}: interrupted", file=sys.stderr)
        return 130
    except (OSError, TypeError, ValueError) as exc:
        message = str(exc).splitlines()[0] if str(exc) else _error_code(exc)
        print(f"{prog}: {message}", file=sys.stderr)
        return 2
    suitable = all(row["suitability"]["capture_suitable"] for row in report["profiles"])
    if report["aborted_reason"]:
        return 2
    return 0 if suitable else 1


__all__ = [
    "REPORT_SCHEMA",
    "REPORT_VERSION",
    "run_system_profile_probe",
    "write_report",
]
