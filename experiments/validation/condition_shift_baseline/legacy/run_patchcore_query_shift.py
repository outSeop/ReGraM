from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
PATCHCORE_SRC = REPO_ROOT / "external" / "patchcore-inspection.clean" / "src"
if str(PATCHCORE_SRC) not in sys.path:
    sys.path.insert(0, str(PATCHCORE_SRC))


patchcore = None
IMAGENET_MEAN = None
IMAGENET_STD = None
DatasetSplit = None
MVTecDataset = None


def lazy_imports() -> None:
    global patchcore, IMAGENET_MEAN, IMAGENET_STD, DatasetSplit, MVTecDataset

    if patchcore is not None:
        return

    import patchcore.backbones as _backbones
    import patchcore.common as _common
    import patchcore.patchcore as _patchcore_module
    import patchcore.sampler as _sampler
    from patchcore.datasets.mvtec import (
        IMAGENET_MEAN as _IMAGENET_MEAN,
        IMAGENET_STD as _IMAGENET_STD,
        DatasetSplit as _DatasetSplit,
        MVTecDataset as _MVTecDataset,
    )

    class _PatchcoreNamespace:
        backbones = _backbones
        common = _common
        patchcore = _patchcore_module
        sampler = _sampler

    patchcore = _PatchcoreNamespace()
    IMAGENET_MEAN = _IMAGENET_MEAN
    IMAGENET_STD = _IMAGENET_STD
    DatasetSplit = _DatasetSplit
    MVTecDataset = _MVTecDataset


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, resize: int = 256, imagesize: int = 224):
        self.root = root
        self.image_paths = sorted(root.glob("*.png"))
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.imagesize = (3, imagesize, imagesize)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        return {
            "image": image,
            "mask": torch.zeros([1, *image.size()[1:]]),
            "classname": self.root.parent.name,
            "anomaly": "good",
            "is_anomaly": 0,
            "image_name": image_path.name,
            "image_path": str(image_path),
        }


def build_patchcore(
    device: torch.device, imagesize: int, num_workers: int, sampler_percentage: float
):
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = None

    nn_method = patchcore.common.FaissNN(False, num_workers)
    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
        sampler_percentage, device
    )

    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, imagesize, imagesize),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=sampler,
        anomaly_score_num_nn=1,
        nn_method=nn_method,
    )
    return model


def make_loader(dataset, batch_size: int, num_workers: int):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader


def summarize_scores(scores: list[float], threshold: float):
    array = np.asarray(scores, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "fpr_over_clean_max": float((array > threshold).mean()) if array.size else 0.0,
    }


def now_kst_string() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_progress(
    progress_path: Path,
    *,
    stage: str,
    category: str,
    threshold_policy: str,
    total_steps: int,
    completed_steps: int,
    current_aug_type: str | None = None,
    current_severity: str | None = None,
    completed_items: list[str] | None = None,
    note: str | None = None,
) -> None:
    lines = [
        "# PatchCore Progress",
        "",
        f"- updated_at: `{now_kst_string()}`",
        f"- category: `{category}`",
        f"- stage: `{stage}`",
        f"- threshold_policy: `{threshold_policy}`",
        f"- completed_steps: `{completed_steps} / {total_steps}`",
    ]
    if current_aug_type is not None:
        lines.append(f"- current_augmentation: `{current_aug_type}`")
    if current_severity is not None:
        lines.append(f"- current_severity: `{current_severity}`")
    if note is not None:
        lines.append(f"- note: `{note}`")
    lines.extend(["", "## Completed Items", ""])
    for item in completed_items or []:
        lines.append(f"- `{item}`")
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument("--aug-root", default="data/query_normal_augmented")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampler-percentage", type=float, default=0.001)
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/runs/patchcore_query_shift",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output / args.category
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "PROGRESS.md"
    partial_results_path = output_dir / "partial_results.json"
    results_path = output_dir / "results.json"

    write_progress(
        progress_path,
        stage="startup",
        category=args.category,
        threshold_policy="clean_max",
        total_steps=0,
        completed_steps=0,
        completed_items=[],
        note="starting_process",
    )
    lazy_imports()

    device = torch.device("cpu")
    category_root = REPO_ROOT / args.data_root / args.category

    train_dataset = MVTecDataset(
        str(REPO_ROOT / args.data_root),
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TRAIN,
    )
    clean_dataset = ImageFolderDataset(
        category_root / "test" / "good", resize=args.resize, imagesize=args.imagesize
    )

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers)
    clean_loader = make_loader(clean_dataset, args.batch_size, args.num_workers)
    aug_root = REPO_ROOT / args.aug_root
    aug_pairs = [
        (aug_type, severity)
        for aug_type in sorted(path.name for path in aug_root.iterdir() if path.is_dir())
        for severity in sorted(
            path.name for path in (aug_root / aug_type).iterdir() if path.is_dir()
        )
    ]
    total_steps = 2 + len(aug_pairs)
    completed_items: list[str] = []

    write_progress(
        progress_path,
        stage="build_model",
        category=args.category,
        threshold_policy="clean_max",
        total_steps=total_steps,
        completed_steps=0,
        completed_items=completed_items,
    )
    model = build_patchcore(
        device, args.imagesize, args.num_workers, args.sampler_percentage
    )

    write_progress(
        progress_path,
        stage="fit_train_good",
        category=args.category,
        threshold_policy="clean_max",
        total_steps=total_steps,
        completed_steps=0,
        completed_items=completed_items,
        note=f"train_good_count={len(train_dataset)}",
    )
    model.fit(train_loader)
    completed_items.append("fit_train_good")

    write_progress(
        progress_path,
        stage="predict_clean_good",
        category=args.category,
        threshold_policy="clean_max",
        total_steps=total_steps,
        completed_steps=1,
        completed_items=completed_items,
        note=f"clean_good_count={len(clean_dataset)}",
    )
    clean_scores, _, _, _ = model.predict(clean_loader)
    clean_threshold = float(np.max(clean_scores))
    completed_items.append("predict_clean_good")

    results = {
        "category": args.category,
        "threshold_policy": "clean_max",
        "updated_at": now_kst_string(),
        "sampler_percentage": args.sampler_percentage,
        "clean_good": summarize_scores(clean_scores, clean_threshold),
        "augmentations": {},
    }
    write_json(partial_results_path, results)

    for aug_index, (aug_type, severity) in enumerate(aug_pairs, start=1):
        if aug_type not in results["augmentations"]:
            results["augmentations"][aug_type] = {}
        write_progress(
            progress_path,
            stage=f"predict_{aug_type}_{severity}",
            category=args.category,
            threshold_policy="clean_max",
            total_steps=total_steps,
            completed_steps=2 + aug_index - 1,
            current_aug_type=aug_type,
            current_severity=severity,
            completed_items=completed_items,
        )
        aug_dataset = ImageFolderDataset(
            aug_root / aug_type / severity / args.category,
            resize=args.resize,
            imagesize=args.imagesize,
        )
        aug_loader = make_loader(aug_dataset, args.batch_size, args.num_workers)
        aug_scores, _, _, _ = model.predict(aug_loader)
        summary = summarize_scores(aug_scores, clean_threshold)
        summary["mean_score_shift"] = summary["mean"] - results["clean_good"]["mean"]
        results["augmentations"][aug_type][severity] = summary
        results["updated_at"] = now_kst_string()
        completed_items.append(f"{aug_type}/{severity}")
        write_json(partial_results_path, results)

    write_json(
        results_path,
        results,
    )
    write_progress(
        progress_path,
        stage="done",
        category=args.category,
        threshold_policy="clean_max",
        total_steps=total_steps,
        completed_steps=total_steps,
        completed_items=completed_items,
        note=f"result={results_path}",
    )
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
