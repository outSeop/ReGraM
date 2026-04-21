from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def merge_masks(mask_dir: Path, output_path: Path) -> None:
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask PNGs found in {mask_dir}")

    merged = None
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file).convert("L")) > 0
        merged = mask if merged is None else np.logical_or(merged, mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((merged.astype(np.uint8) * 255), mode="L").save(output_path)


def build_meta(root: Path) -> dict:
    info: dict[str, dict[str, list[dict]]] = {"train": {}, "test": {}}
    for cls_name in CLASSES:
        cls_root = root / cls_name
        for phase in ("train", "test"):
            cls_info = []
            for specie in sorted(os.listdir(cls_root / phase)):
                is_abnormal = specie != "good"
                img_dir = cls_root / phase / specie
                img_names = sorted(os.listdir(img_dir))
                if is_abnormal:
                    mask_dir = cls_root / "ground_truth_merge_mask" / f"{specie}_merge_mask"
                    mask_names = sorted(os.listdir(mask_dir))
                else:
                    mask_names = []

                for idx, img_name in enumerate(img_names):
                    cls_info.append(
                        {
                            "img_path": f"{cls_name}/{phase}/{specie}/{img_name}",
                            "mask_path": (
                                f"{cls_name}/ground_truth_merge_mask/{specie}_merge_mask/{mask_names[idx]}"
                                if is_abnormal
                                else ""
                            ),
                            "cls_name": cls_name,
                            "specie_name": specie,
                            "anomaly": 1 if is_abnormal else 0,
                        }
                    )
            info[phase][cls_name] = cls_info
    return info


def prepare_dataset(src_root: Path, dst_root: Path) -> None:
    for cls_name in CLASSES:
        src_cls = src_root / cls_name
        dst_cls = dst_root / cls_name
        ensure_symlink(src_cls / "train", dst_cls / "train")
        ensure_symlink(src_cls / "test", dst_cls / "test")
        ensure_symlink(src_cls / "validation", dst_cls / "validation")

        gt_root = src_cls / "ground_truth"
        for anomaly_type in sorted(os.listdir(gt_root)):
            anomaly_dir = gt_root / anomaly_type
            output_dir = dst_cls / "ground_truth_merge_mask" / f"{anomaly_type}_merge_mask"
            for sample_dir in sorted(anomaly_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                output_path = output_dir / f"{sample_dir.name}_mask.png"
                if not output_path.exists():
                    merge_masks(sample_dir, output_path)

    meta = build_meta(dst_root)
    (dst_root / "meta.json").write_text(json.dumps(meta, indent=4), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument("--dst-root", default="data/mvtec_loco_caption")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    prepare_dataset(src_root, dst_root)
    print(dst_root.resolve())


if __name__ == "__main__":
    main()
