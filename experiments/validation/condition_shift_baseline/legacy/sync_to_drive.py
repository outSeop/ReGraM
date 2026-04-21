from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyncItem:
    src: str
    kind: str = "any"  # any, file, dir


PROFILES: dict[str, list[SyncItem]] = {
    "notebook_assets": [
        SyncItem("experiments/validation/condition_shift_baseline/notebook/experiment.ipynb", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/reports", "dir"),
        SyncItem("experiments/validation/condition_shift_baseline/colab", "dir"),
        SyncItem("experiments/validation/condition_shift_baseline/src/core/contracts.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/univad/prepare_mvtec_loco.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/univad/prepare_smoke_subset.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/univad/run_smoke_colab.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/univad/run_clean_eval.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/core/run_patchcore_manifest_shift.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/core/augmentation_runtime.py", "file"),
    ],
    "patchcore_results": [
        SyncItem("experiments/validation/condition_shift_baseline/reports/patchcore_manifest_shift", "dir"),
        SyncItem("experiments/validation/condition_shift_baseline/reports/patchcore_clean_eval_breakfast_box.json", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/reports/patchcore_identity_repro", "dir"),
        SyncItem("experiments/validation/condition_shift_baseline/reports/sample_panels", "dir"),
    ],
    "univad_runtime": [
        SyncItem("external/UniVAD", "dir"),
        SyncItem(".venv_univad311", "dir"),
        SyncItem("experiments/validation/condition_shift_baseline/colab", "dir"),
        SyncItem("experiments/validation/condition_shift_baseline/src/univad/prepare_mvtec_loco.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/univad/prepare_smoke_subset.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/univad/run_smoke_colab.py", "file"),
        SyncItem("experiments/validation/condition_shift_baseline/src/univad/run_clean_eval.py", "file"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy selected ReGraM experiment assets into a Google Drive synced folder."
    )
    parser.add_argument(
        "--drive-root",
        required=True,
        help="Local Google Drive synced folder root, e.g. ~/Library/CloudStorage/GoogleDrive-.../My Drive",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Local ReGraM repo root.",
    )
    parser.add_argument(
        "--dest-subdir",
        default="ReGraM",
        help="Destination subdirectory inside drive-root.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILES.keys()),
        help="Named sync profile. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional relative paths to copy.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Relative paths to exclude after profile expansion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete destination path before copying each selected item.",
    )
    return parser.parse_args()


def ensure_expected_type(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if kind == "file" and not path.is_file():
        raise ValueError(f"Expected file: {path}")
    if kind == "dir" and not path.is_dir():
        raise ValueError(f"Expected directory: {path}")


def expand_items(repo_root: Path, profiles: list[str], includes: list[str], excludes: set[str]) -> list[Path]:
    selected: dict[str, str] = {}
    for profile in profiles:
        for item in PROFILES[profile]:
            selected[item.src] = item.kind
    for rel in includes:
        selected[rel] = "any"

    for rel in excludes:
        selected.pop(rel, None)

    paths: list[Path] = []
    for rel, kind in selected.items():
        src = repo_root / rel
        ensure_expected_type(src, kind)
        paths.append(src)
    return sorted(paths, key=lambda p: str(p))


def copy_path(src: Path, repo_root: Path, dest_root: Path, clean: bool, dry_run: bool) -> dict[str, str]:
    rel = src.relative_to(repo_root)
    dst = dest_root / rel

    if dry_run:
        return {"src": str(src), "dst": str(dst), "status": "planned"}

    dst.parent.mkdir(parents=True, exist_ok=True)
    if clean and dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

    return {"src": str(src), "dst": str(dst), "status": "copied"}


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    drive_root = Path(args.drive_root).expanduser().resolve()
    dest_root = drive_root / args.dest_subdir

    profiles = args.profile or ["notebook_assets"]
    excludes = set(args.exclude)
    paths = expand_items(repo_root, profiles, args.include, excludes)

    operations = [
        copy_path(src=path, repo_root=repo_root, dest_root=dest_root, clean=args.clean, dry_run=args.dry_run)
        for path in paths
    ]

    manifest = {
        "repo_root": str(repo_root),
        "drive_root": str(drive_root),
        "dest_root": str(dest_root),
        "profiles": profiles,
        "dry_run": args.dry_run,
        "clean": args.clean,
        "operations": operations,
    }

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
