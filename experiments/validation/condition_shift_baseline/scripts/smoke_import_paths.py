"""Smoke-check import paths for refactor compatibility.

Usage:
    python experiments/validation/condition_shift_baseline/scripts/smoke_import_paths.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main() -> int:
    exp_root = Path(__file__).resolve().parents[1]
    src_root = exp_root / "src"
    core_root = src_root / "core"

    for path in (src_root, core_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    modules = [
        "common.contracts",
        "common.repo_paths",
        "data.manifest_paths",
        "core.contracts",
        "core.manifest_paths",
    ]

    for name in modules:
        importlib.import_module(name)
        print(f"[ok] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
