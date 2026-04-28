"""Run refactor validation checks in one command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]

    run(
        [
            "python",
            "experiments/validation/condition_shift_baseline/scripts/smoke_import_paths.py",
        ],
        cwd=repo_root,
    )
    run(
        [
            "pytest",
            "-q",
            "experiments/validation/condition_shift_baseline/tests/test_import_compatibility.py",
        ],
        cwd=repo_root,
    )

    # Full manifest-shift helper test needs Pillow available in the runtime.
    try:
        __import__("PIL")
    except ModuleNotFoundError:
        print("[warn] skip test_manifest_shift_common.py (Pillow not installed)")
        return 0

    run(
        [
            "pytest",
            "-q",
            "experiments/validation/condition_shift_baseline/tests/test_manifest_shift_common.py",
        ],
        cwd=repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
