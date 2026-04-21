from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def build_smoke_meta(class_name: str) -> dict:
    return {
        "train": {
            class_name: [
                {
                    "img_path": f"{class_name}/train/good/000.png",
                    "mask_path": "",
                    "cls_name": class_name,
                    "specie_name": "good",
                    "anomaly": 0,
                }
            ]
        },
        "test": {
            class_name: [
                {
                    "img_path": f"{class_name}/test/good/000.png",
                    "mask_path": "",
                    "cls_name": class_name,
                    "specie_name": "good",
                    "anomaly": 0,
                },
                {
                    "img_path": f"{class_name}/test/logical_anomalies/000.png",
                    "mask_path": f"{class_name}/ground_truth_merge_mask/logical_anomalies_merge_mask/000_mask.png",
                    "cls_name": class_name,
                    "specie_name": "logical_anomalies",
                    "anomaly": 1,
                },
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default="data/mvtec_loco_caption")
    parser.add_argument("--dst-root", default="data/mvtec_loco_caption_smoke")
    parser.add_argument("--class-name", default="breakfast_box")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    class_name = args.class_name

    src_cls = src_root / class_name
    dst_cls = dst_root / class_name

    ensure_link_or_copy(src_cls / "train", dst_cls / "train")
    ensure_link_or_copy(src_cls / "test", dst_cls / "test")
    ensure_link_or_copy(src_cls / "validation", dst_cls / "validation")
    ensure_link_or_copy(src_cls / "ground_truth_merge_mask", dst_cls / "ground_truth_merge_mask")

    dst_root.mkdir(parents=True, exist_ok=True)
    meta = build_smoke_meta(class_name)
    (dst_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(dst_root.resolve())


if __name__ == "__main__":
    main()
