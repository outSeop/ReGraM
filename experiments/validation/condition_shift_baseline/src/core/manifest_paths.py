"""Path helpers for manifest-backed runners."""

from __future__ import annotations

from pathlib import Path

from repo_paths import REPO_ROOT


def resolve_manifest_image_path(entry: dict) -> Path:
    source_path = Path(entry["source_path"])
    path_mode = entry.get("source_path_mode", "absolute")
    if source_path.is_absolute():
        return source_path
    if path_mode == "repo_relative":
        return REPO_ROOT / source_path
    return source_path.resolve()
