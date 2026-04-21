from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy prepared datasets or small report assets into an existing runtime repo."
    )
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--drive-repo-name", default="ReGraM")
    parser.add_argument("--runtime-repo-root", default="/content/ReGraM")
    parser.add_argument("--copy-reports", action="store_true")
    parser.add_argument("--prepared-dataset", action="append", default=["mvtec_loco_caption", "mvtec_loco_caption_smoke"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def copy_path(src: Path, dst: Path, dry_run: bool) -> dict[str, str]:
    if not src.exists():
        return {"status": "missing", "src": str(src), "dst": str(dst)}

    if dry_run:
        return {"status": "planned", "src": str(src), "dst": str(dst)}

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return {"status": "copied", "src": str(src), "dst": str(dst)}


def main() -> None:
    args = parse_args()
    drive_root = Path(args.drive_root).expanduser().resolve()
    drive_repo_root = drive_root / args.drive_repo_name
    runtime_repo_root = Path(args.runtime_repo_root).resolve()

    operations: list[dict[str, str]] = []

    for dataset_name in args.prepared_dataset:
        operations.append(
            copy_path(
                src=drive_repo_root / "data" / dataset_name,
                dst=runtime_repo_root / "data" / dataset_name,
                dry_run=args.dry_run,
            )
        )

    if args.copy_reports:
        operations.append(
            copy_path(
                src=drive_repo_root / "experiments" / "validation" / "condition_shift_baseline" / "reports",
                dst=runtime_repo_root / "experiments" / "validation" / "condition_shift_baseline" / "reports",
                dry_run=args.dry_run,
            )
        )

    manifest = {
        "drive_repo_root": str(drive_repo_root),
        "runtime_repo_root": str(runtime_repo_root),
        "dry_run": args.dry_run,
        "operations": operations,
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
