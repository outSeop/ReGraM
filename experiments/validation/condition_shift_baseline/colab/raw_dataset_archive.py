from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or restore a raw dataset tar archive for Colab runtime use."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser(
        "pack",
        help="Create a tar archive from an existing dataset directory.",
    )
    pack_parser.add_argument(
        "--src-root",
        default="/content/drive/MyDrive/ReGraM/data/row/mvtec_loco_anomaly_detection",
    )
    pack_parser.add_argument(
        "--archive-path",
        default="/content/drive/MyDrive/ReGraM/data/row/mvtec_loco_anomaly_detection.tar",
    )
    pack_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination archive if it already exists.",
    )

    restore_parser = subparsers.add_parser(
        "restore",
        help="Copy a tar archive to runtime storage and extract it into the repo data root.",
    )
    restore_parser.add_argument(
        "--archive-path",
        default="/content/drive/MyDrive/ReGraM/data/row/mvtec_loco_anomaly_detection.tar",
    )
    restore_parser.add_argument(
        "--runtime-archive-path",
        default="/content/mvtec_loco_anomaly_detection.tar",
    )
    restore_parser.add_argument(
        "--extract-root",
        default="/content/ReGraM/data/row",
    )
    restore_parser.add_argument(
        "--dataset-dirname",
        default="mvtec_loco_anomaly_detection",
    )
    restore_parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Assume the archive is already present at runtime-archive-path.",
    )
    restore_parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Delete any existing extracted dataset directory and extract again.",
    )

    return parser


def pack_dataset(src_root: Path, archive_path: Path, force: bool) -> dict[str, object]:
    if not src_root.exists():
        raise FileNotFoundError(f"Dataset source root not found: {src_root}")
    if archive_path.exists() and not force:
        raise FileExistsError(f"Archive already exists: {archive_path}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    started_at = time.time()
    with tarfile.open(archive_path, "w") as tar:
        tar.add(src_root, arcname=src_root.name)

    return {
        "command": "pack",
        "src_root": str(src_root),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }


def restore_dataset(
    archive_path: Path,
    runtime_archive_path: Path,
    extract_root: Path,
    dataset_dirname: str,
    skip_copy: bool,
    force_extract: bool,
) -> dict[str, object]:
    if not archive_path.exists() and not skip_copy:
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    extract_root.mkdir(parents=True, exist_ok=True)
    extracted_root = extract_root / dataset_dirname

    operations: list[dict[str, str]] = []
    started_at = time.time()

    if not skip_copy:
        runtime_archive_path.parent.mkdir(parents=True, exist_ok=True)
        if runtime_archive_path.exists():
            runtime_archive_path.unlink()
        shutil.copy2(archive_path, runtime_archive_path)
        operations.append(
            {
                "step": "copied_archive",
                "src": str(archive_path),
                "dst": str(runtime_archive_path),
            }
        )

    if extracted_root.exists() and force_extract:
        shutil.rmtree(extracted_root)
        operations.append(
            {
                "step": "removed_existing_extract",
                "path": str(extracted_root),
            }
        )

    if not extracted_root.exists():
        with tarfile.open(runtime_archive_path, "r:*") as tar:
            tar.extractall(extract_root)
        operations.append(
            {
                "step": "extracted_archive",
                "archive_path": str(runtime_archive_path),
                "extract_root": str(extract_root),
            }
        )
    else:
        operations.append(
            {
                "step": "reused_existing_extract",
                "path": str(extracted_root),
            }
        )

    return {
        "command": "restore",
        "archive_path": str(archive_path),
        "runtime_archive_path": str(runtime_archive_path),
        "extract_root": str(extract_root),
        "dataset_root": str(extracted_root),
        "dataset_exists": extracted_root.exists(),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "operations": operations,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pack":
        result = pack_dataset(
            src_root=Path(args.src_root).expanduser().resolve(),
            archive_path=Path(args.archive_path).expanduser().resolve(),
            force=args.force,
        )
    else:
        result = restore_dataset(
            archive_path=Path(args.archive_path).expanduser().resolve(),
            runtime_archive_path=Path(args.runtime_archive_path).expanduser().resolve(),
            extract_root=Path(args.extract_root).expanduser().resolve(),
            dataset_dirname=args.dataset_dirname,
            skip_copy=args.skip_copy,
            force_extract=args.force_extract,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
