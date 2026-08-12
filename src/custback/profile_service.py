"""Restart-aware system profile staging and public state projection."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import (
    AppConfig,
    ConfigVersionConflictError,
    RuntimeConfig,
    changed_config_paths,
)
from .profile_preferences import (
    PreferenceRevisionConflict,
    ProfilePreferences,
    ProfilePreferencesError,
    ProfilePreferencesStore,
    apply_preferences,
)
from .system_profiles import (
    NONQUALIFIED_EVIDENCE_STATES,
    PROFILE_CATALOG,
    ProfileCatalog,
    ProfileCatalogError,
    delete_path,
    flatten_patch,
    matching_profile,
    profile_availability,
    public_catalog,
    set_path,
)


class ProfileServiceError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ProfileStartupContext:
    base_config: AppConfig
    preferences: ProfilePreferences
    store: ProfilePreferencesStore | None
    cli_values: Mapping[str, Any]

    @property
    def cli_locks(self) -> frozenset[str]:
        return frozenset(self.cli_values)


def _apply_flat(config: AppConfig, values: Mapping[str, Any]) -> AppConfig:
    raw = config.to_dict()
    for path, value in values.items():
        set_path(raw, path, value)
    return AppConfig.from_dict(raw)


class ProfileService:
    """Own profile matching and durable staging without lifecycle privilege."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        startup: ProfileStartupContext,
        *,
        runtime_facts: Callable[[], Mapping[str, Any]] | None = None,
        catalog: ProfileCatalog = PROFILE_CATALOG,
    ) -> None:
        self.runtime = runtime
        self.startup = startup
        self.catalog = catalog
        self._runtime_facts = runtime_facts or (lambda: {})

    def _read_preferences(self) -> ProfilePreferences:
        if self.startup.store is None:
            return self.startup.preferences
        return self.startup.store.read()

    def _desired_config(self, preferences: ProfilePreferences) -> AppConfig:
        desired = apply_preferences(self.startup.base_config, preferences)
        return _apply_flat(desired, self.startup.cli_values)

    def _facts(self) -> dict[str, Any]:
        try:
            facts = dict(self._runtime_facts())
        except Exception:
            facts = {}
        return facts

    def _availability(
        self,
        definition,
        candidate: AppConfig,
        facts: Mapping[str, Any],
    ) -> tuple[bool, str]:
        available, reason = profile_availability(definition, candidate, facts)
        if not available:
            return available, reason
        flattened = flatten_patch(definition.detached_patch())
        for path in definition.owned_paths:
            if (
                path in self.startup.cli_locks
                and self.startup.cli_values[path] != flattened[path]
            ):
                return False, f"Explicit CLI override locks {path}."
        return True, ""

    def _effective_match(
        self,
        definition,
        config: AppConfig,
        facts: Mapping[str, Any],
    ) -> bool:
        available, _ = self._availability(definition, config, facts)
        if not available:
            return False
        mode = definition.requirements.get("output_mode")
        if mode:
            if not all(
                key in facts for key in ("output_width", "output_height", "output_fps")
            ):
                return False
            actual = (
                facts.get("output_width"),
                facts.get("output_height"),
                facts.get("output_fps"),
            )
            if tuple(mode) != actual:
                return False
        provider = definition.requirements.get("provider")
        if provider:
            active = str(facts.get("acceleration_active_provider", "")).lower()
            selection = facts.get("segmentation_selection")
            selected_provider = (
                str(selection.get("active_provider", "")).lower()
                if isinstance(selection, Mapping)
                else ""
            )
            if provider not in {active, selected_provider}:
                return False
        return True

    def status(self) -> dict[str, Any]:
        state = self.runtime.read()
        preferences = self._read_preferences()
        desired = self._desired_config(preferences)
        startup_desired = self._desired_config(self.startup.preferences)
        facts = self._facts()
        catalog = public_catalog(self.catalog)
        axes_status: dict[str, Any] = {}
        for axis, definitions in self.catalog.axes.items():
            for identifier, definition in definitions.items():
                available, reason = self._availability(definition, desired, facts)
                public = catalog["axes"][axis]["profiles"][identifier]
                public["available"] = available
                public["availability_reason"] = reason
            active = matching_profile(state.config, axis, self.catalog)
            desired_match = matching_profile(desired, axis, self.catalog)
            desired_available = False
            if desired_match is not None:
                desired_available, _ = self._availability(
                    definitions[desired_match], desired, facts
                )
            active_effective = bool(
                active is not None
                and self._effective_match(definitions[active], state.config, facts)
            )
            pending = sorted(
                path
                for path in changed_config_paths(startup_desired, desired)
                if path in self.catalog.owned_paths[axis]
            )
            if desired_match is not None and (
                not desired_available
                or (active == desired_match and not active_effective)
            ):
                profile_state = "configured_unavailable"
            elif pending:
                profile_state = "saved_for_restart"
            elif active == desired_match and active is not None:
                profile_state = "active"
            else:
                profile_state = "custom"
            axes_status[axis] = {
                "state": profile_state,
                "active": active,
                "desired": desired_match,
                "pending_restart_fields": pending,
            }
        return {
            "schema": "custback.system-profiles-status",
            "version": 1,
            "catalog": catalog,
            "config_version": state.version,
            "preference_revision": preferences.revision,
            "cli_locks": sorted(self.startup.cli_locks & self.catalog.manageable_paths),
            "pending_restart_fields": sorted(
                path
                for path in changed_config_paths(startup_desired, desired)
                if path in self.catalog.manageable_paths
            ),
            "axes": axes_status,
        }

    @staticmethod
    def _validate_config_revision(expected_config_version: int) -> None:
        if type(expected_config_version) is not int or expected_config_version < 0:
            raise ProfileServiceError(
                422, "invalid_revision", "config revision is invalid"
            )

    @staticmethod
    def _config_conflict(exc: ConfigVersionConflictError) -> ProfileServiceError:
        return ProfileServiceError(
            409,
            "stale_config_version",
            f"configuration changed concurrently; current version is "
            f"{exc.current_version}",
        )

    def apply(
        self,
        selections: Mapping[str, str],
        *,
        expected_config_version: int,
        expected_preferences_revision: int,
        accept_experimental: bool,
    ) -> dict[str, Any]:
        if self.startup.store is None:
            raise ProfileServiceError(
                409, "preferences_disabled", "profile preferences are disabled"
            )
        if not isinstance(selections, Mapping) or not selections:
            raise ProfileServiceError(
                422, "invalid_selection", "at least one profile selection is required"
            )
        if set(selections) - set(self.catalog.axes):
            raise ProfileServiceError(
                422, "invalid_selection", "profile axis is unknown"
            )
        definitions = {}
        for axis, identifier in selections.items():
            if not isinstance(identifier, str):
                raise ProfileServiceError(
                    422, "invalid_selection", "profile ID is invalid"
                )
            try:
                definition = self.catalog.definition(axis, identifier)
            except ProfileCatalogError as exc:
                raise ProfileServiceError(422, "invalid_selection", str(exc)) from exc
            if (
                definition.evidence_state in NONQUALIFIED_EVIDENCE_STATES
                and accept_experimental is not True
            ):
                raise ProfileServiceError(
                    409,
                    "experimental_ack_required",
                    "experimental profiles require explicit acknowledgment",
                )
            definitions[axis] = definition
        self._validate_config_revision(expected_config_version)
        facts = self._facts()

        def mutate(values: dict[str, Any]) -> dict[str, Any]:
            candidate_values = copy.deepcopy(values)
            for axis, definition in definitions.items():
                for path in self.catalog.owned_paths[axis]:
                    delete_path(candidate_values, path)
                for path, value in flatten_patch(definition.detached_patch()).items():
                    if path in self.startup.cli_locks:
                        locked = self.startup.cli_values[path]
                        if locked != value:
                            raise ProfileServiceError(
                                409,
                                "cli_override_conflict",
                                f"profile selection conflicts with locked CLI field {path}",
                            )
                    set_path(candidate_values, path, value)
            try:
                proposed = self._desired_config(
                    ProfilePreferences(
                        expected_preferences_revision + 1, candidate_values
                    )
                )
            except (TypeError, ValueError, ProfilePreferencesError) as exc:
                raise ProfileServiceError(
                    409,
                    "profile_unavailable",
                    "The profile conflicts with the operator configuration.",
                ) from exc
            for definition in definitions.values():
                available, reason = self._availability(definition, proposed, facts)
                if not available:
                    raise ProfileServiceError(409, "profile_unavailable", reason)
            return candidate_values

        try:
            with self.runtime.guard_version(expected_config_version):
                self.startup.store.update(expected_preferences_revision, mutate)
        except ConfigVersionConflictError as exc:
            raise self._config_conflict(exc) from exc
        except PreferenceRevisionConflict as exc:
            raise ProfileServiceError(
                409,
                "stale_preference_revision",
                f"profile preferences changed concurrently; current revision is {exc.current}",
            ) from exc
        except ProfilePreferencesError as exc:
            raise ProfileServiceError(409, "preferences_error", str(exc)) from exc
        return self.status()

    def reset(
        self,
        axes: list[str],
        *,
        expected_config_version: int,
        expected_preferences_revision: int,
    ) -> dict[str, Any]:
        if self.startup.store is None:
            raise ProfileServiceError(
                409, "preferences_disabled", "profile preferences are disabled"
            )
        if (
            not isinstance(axes, list)
            or not axes
            or len(axes) != len(set(axes))
            or any(axis not in self.catalog.axes for axis in axes)
        ):
            raise ProfileServiceError(
                422, "invalid_selection", "reset axes are invalid"
            )
        self._validate_config_revision(expected_config_version)

        def mutate(values: dict[str, Any]) -> dict[str, Any]:
            for axis in axes:
                for path in self.catalog.owned_paths[axis]:
                    delete_path(values, path)
            return values

        try:
            with self.runtime.guard_version(expected_config_version):
                self.startup.store.update(expected_preferences_revision, mutate)
        except ConfigVersionConflictError as exc:
            raise self._config_conflict(exc) from exc
        except PreferenceRevisionConflict as exc:
            raise ProfileServiceError(
                409,
                "stale_preference_revision",
                f"profile preferences changed concurrently; current revision is {exc.current}",
            ) from exc
        except ProfilePreferencesError as exc:
            raise ProfileServiceError(409, "preferences_error", str(exc)) from exc
        return self.status()


__all__ = [
    "ProfileService",
    "ProfileServiceError",
    "ProfileStartupContext",
]
