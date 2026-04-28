from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_refactor_import_compatibility_paths() -> None:
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
    for module_name in modules:
        importlib.import_module(module_name)
