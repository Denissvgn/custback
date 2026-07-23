"""Per-platform user-directory routing.

This is the path half of the platform seam: callers ask for a semantic
directory instead of embedding XDG or Windows Known-Folder conventions.  The
helpers are deliberately side-effect free; creating and securing a directory
remains the responsibility of the storage/token caller that uses it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


def _resolved_home(
    environ: Mapping[str, str],
    home: Path | None,
    *,
    windows: bool,
) -> Path:
    if home is not None:
        return Path(home)
    variable = "USERPROFILE" if windows else "HOME"
    configured = environ.get(variable, "")
    if configured:
        return Path(configured)
    return Path.home()


def config_dir(
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the current user's Custback configuration/token directory.

    Windows uses the roaming Application Data Known Folder (surfaced to Python
    as ``APPDATA``). POSIX uses an absolute ``XDG_CONFIG_HOME`` when supplied,
    otherwise ``~/.config``. Relative environment overrides are ignored so a
    service's working directory can never silently become configuration
    authority.
    """

    current_platform = sys.platform if platform is None else platform
    env = os.environ if environ is None else environ
    if current_platform == "win32":
        configured = env.get("APPDATA", "")
        base = (
            Path(configured) if configured and Path(configured).is_absolute() else None
        )
        if base is None:
            base = _resolved_home(env, home, windows=True) / "AppData" / "Roaming"
        return base / "Custback"

    configured = env.get("XDG_CONFIG_HOME", "")
    base = Path(configured) if configured and Path(configured).is_absolute() else None
    if base is None:
        base = _resolved_home(env, home, windows=False) / ".config"
    return base / "custback"
