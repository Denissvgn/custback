"""Shared JSON Merge Patch semantics for runtime configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def merge_patch(target: Any, patch: Any) -> Any:
    """Return a deep-copied RFC 7396 merge of ``patch`` into ``target``.

    Objects merge recursively while arrays and scalar values replace the
    corresponding target value. ``null`` removes an existing key, which lets
    model validation restore that field's declared default. A null-valued key
    absent from the target is retained so strict Pydantic models still report
    unknown configuration fields instead of silently accepting them.

    Neither input is mutated and no mutable value from either input is reused
    in the result.
    """

    if not isinstance(patch, dict):
        return deepcopy(patch)

    source = target if isinstance(target, dict) else {}
    merged = deepcopy(source)
    for key, value in patch.items():
        if value is None:
            if key in source:
                merged.pop(key, None)
            else:
                # Pure RFC 7396 treats this as a no-op. Configuration patches
                # retain it so validators can reject unknown keys.
                merged[key] = None
            continue
        merged[key] = merge_patch(source.get(key), value)
    return merged
