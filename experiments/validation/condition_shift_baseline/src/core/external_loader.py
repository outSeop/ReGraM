"""Centralized sys.path registration for external baseline sources."""

from __future__ import annotations

import sys
from pathlib import Path

from repo_paths import REPO_ROOT


EXTERNAL_SOURCES: dict[str, Path] = {
    "patchcore": REPO_ROOT / "external" / "patchcore-inspection.clean" / "src",
    "univad": REPO_ROOT / "external" / "UniVAD",
}


def ensure_external_on_path(name: str) -> Path:
    if name not in EXTERNAL_SOURCES:
        raise KeyError(
            f"Unknown external source '{name}'. Registered: {sorted(EXTERNAL_SOURCES)}"
        )
    source_path = EXTERNAL_SOURCES[name]
    if not source_path.exists():
        raise RuntimeError(
            f"External source '{name}' not found at {source_path}. "
            f"Expected it at external/... relative to repo root. "
            f"Update external_loader.EXTERNAL_SOURCES if the path changed."
        )
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    return source_path
