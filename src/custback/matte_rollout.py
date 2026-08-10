"""Fail-closed MATTE-5.4 rollout state and sanitized counters."""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping
from typing import Any, Literal

from .config import AppConfig, legacy_matte_policy_patch, uses_legacy_matte_policy


MATTE_ROLLOUT_SCHEMA = "custback.matte-rollout-status"
MATTE_ROLLOUT_VERSION = 1
MATTE_ROLLOUT_STAGE = "compatibility_hold"
MATTE_ROLLOUT_DECISION = "held_pending_physical_qualification"
MATTE_PRESET_CATALOG_VERSION = 1
MATTE_PRESET_EVIDENCE_STATUS = "not_qualified"
LEGACY_MATTE_ROLLBACK_PATCH_ID = "matte-legacy-v1"

MattePatchKind = Literal["matte_update", "legacy_rollback"]
MatteRolloutOutcome = Literal["none", "attempt", "success", "failure", "rollback"]

_MATTE_PATCH_SECTIONS = frozenset({"segmentation", "acceleration", "compositing"})
_ROLLOUT_STATUS_KEYS = frozenset(
    {
        "schema",
        "version",
        "stage",
        "decision",
        "configured_schema_version",
        "config_version",
        "qualified_default_active",
        "preset_catalog_version",
        "preset_evidence_status",
        "legacy_policy_available",
        "legacy_policy_active",
        "rollback_patch_id",
        "patch_attempts",
        "patch_in_flight",
        "patch_successes",
        "patch_failures",
        "legacy_rollbacks",
        "last_outcome",
    }
)
_OUTCOMES = frozenset({"none", "attempt", "success", "failure", "rollback"})
_MAX_COUNTER = 2**63 - 1


def classify_matte_patch(patch: object) -> MattePatchKind | None:
    """Classify a merge patch without retaining or exposing any patch values."""

    if not isinstance(patch, dict) or not (_MATTE_PATCH_SECTIONS & patch.keys()):
        return None
    if patch == legacy_matte_policy_patch():
        return "legacy_rollback"
    return "matte_update"


def empty_matte_rollout_status() -> dict[str, object]:
    """Return valid pre-start status with no process-local rollout activity."""

    return {
        "schema": MATTE_ROLLOUT_SCHEMA,
        "version": MATTE_ROLLOUT_VERSION,
        "stage": MATTE_ROLLOUT_STAGE,
        "decision": MATTE_ROLLOUT_DECISION,
        "configured_schema_version": 1,
        "config_version": 0,
        "qualified_default_active": False,
        "preset_catalog_version": MATTE_PRESET_CATALOG_VERSION,
        "preset_evidence_status": MATTE_PRESET_EVIDENCE_STATUS,
        "legacy_policy_available": True,
        "legacy_policy_active": True,
        "rollback_patch_id": LEGACY_MATTE_ROLLBACK_PATCH_ID,
        "patch_attempts": 0,
        "patch_in_flight": 0,
        "patch_successes": 0,
        "patch_failures": 0,
        "legacy_rollbacks": 0,
        "last_outcome": "none",
    }


def validate_matte_rollout_status(value: object) -> dict[str, object]:
    """Validate the exact path-free public rollout status contract."""

    if not isinstance(value, Mapping) or frozenset(value) != _ROLLOUT_STATUS_KEYS:
        raise ValueError("matte_rollout must use the exact version-1 status schema")
    status = dict(value)
    if (
        status["schema"] != MATTE_ROLLOUT_SCHEMA
        or status["version"] != MATTE_ROLLOUT_VERSION
        or status["stage"] != MATTE_ROLLOUT_STAGE
        or status["decision"] != MATTE_ROLLOUT_DECISION
        or status["qualified_default_active"] is not False
        or status["preset_catalog_version"] != MATTE_PRESET_CATALOG_VERSION
        or status["preset_evidence_status"] != MATTE_PRESET_EVIDENCE_STATUS
        or status["legacy_policy_available"] is not True
        or status["rollback_patch_id"] != LEGACY_MATTE_ROLLBACK_PATCH_ID
        or type(status["legacy_policy_active"]) is not bool
        or status["last_outcome"] not in _OUTCOMES
    ):
        raise ValueError("matte_rollout contains an invalid rollout disposition")
    configured_schema_version = status["configured_schema_version"]
    if type(configured_schema_version) is not int or configured_schema_version < 1:
        raise ValueError("matte_rollout configured_schema_version is invalid")
    config_version = status["config_version"]
    if type(config_version) is not int or config_version < 0:
        raise ValueError("matte_rollout config_version is invalid")
    for name in (
        "patch_attempts",
        "patch_in_flight",
        "patch_successes",
        "patch_failures",
        "legacy_rollbacks",
    ):
        counter = status[name]
        if type(counter) is not int or not 0 <= counter <= _MAX_COUNTER:
            raise ValueError(f"matte_rollout {name} is invalid")
    if status["patch_attempts"] != (
        status["patch_successes"] + status["patch_failures"] + status["patch_in_flight"]
    ):
        raise ValueError("matte_rollout attempt counters are inconsistent")
    if status["legacy_rollbacks"] > status["patch_successes"]:
        raise ValueError("matte_rollout rollback count exceeds successes")
    return copy.deepcopy(status)


class MatteRolloutTelemetry:
    """Process-local, content-free counters for matte configuration canaries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts = 0
        self._in_flight = 0
        self._successes = 0
        self._failures = 0
        self._rollbacks = 0
        self._last_outcome: MatteRolloutOutcome = "none"

    def record_attempt(self) -> None:
        with self._lock:
            if self._attempts >= _MAX_COUNTER:
                raise OverflowError("matte rollout attempt counter is exhausted")
            self._attempts += 1
            self._in_flight += 1
            self._last_outcome = "attempt"

    def begin(self, kind: MattePatchKind) -> "MatteRolloutAttempt":
        if kind not in {"matte_update", "legacy_rollback"}:
            raise ValueError("unsupported matte rollout patch kind")
        self.record_attempt()
        return MatteRolloutAttempt(self, kind)

    def record_success(self, kind: MattePatchKind) -> None:
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("matte rollout success has no matching attempt")
            self._in_flight -= 1
            self._successes += 1
            if kind == "legacy_rollback":
                self._rollbacks += 1
                self._last_outcome = "rollback"
            else:
                self._last_outcome = "success"

    def record_failure(self) -> None:
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("matte rollout failure has no matching attempt")
            self._in_flight -= 1
            self._failures += 1
            self._last_outcome = "failure"

    def snapshot(self, config: AppConfig, config_version: int = 0) -> dict[str, object]:
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        if type(config_version) is not int or config_version < 0:
            raise ValueError("config_version must be a nonnegative integer")
        with self._lock:
            status = empty_matte_rollout_status()
            status.update(
                {
                    "configured_schema_version": config.schema_version,
                    "config_version": config_version,
                    "legacy_policy_active": uses_legacy_matte_policy(config),
                    "patch_attempts": self._attempts,
                    "patch_in_flight": self._in_flight,
                    "patch_successes": self._successes,
                    "patch_failures": self._failures,
                    "legacy_rollbacks": self._rollbacks,
                    "last_outcome": self._last_outcome,
                }
            )
        return validate_matte_rollout_status(status)


class MatteRolloutAttempt:
    """One idempotently completed rollout attempt shared with the frame lane."""

    def __init__(self, telemetry: MatteRolloutTelemetry, kind: MattePatchKind) -> None:
        self._telemetry = telemetry
        self._kind: MattePatchKind = kind
        self._lock = threading.Lock()
        self._completed = False

    def succeed(self) -> bool:
        with self._lock:
            if self._completed:
                return False
            self._completed = True
            self._telemetry.record_success(self._kind)
            return True

    def fail(self) -> bool:
        with self._lock:
            if self._completed:
                return False
            self._completed = True
            self._telemetry.record_failure()
            return True


def public_legacy_matte_rollback_patch() -> dict[str, Any]:
    """Named public alias used by documentation and API-level tests."""

    return legacy_matte_policy_patch()
