from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from augmentation_runtime import apply_augmentation, build_manifest_entries, load_manifest
from contracts import build_summary, write_log, write_summary


REPO_ROOT = Path(__file__).resolve().parents[3]
PATCHCORE_SRC = REPO_ROOT / "external" / "patchcore-inspection.clean" / "src"
if str(PATCHCORE_SRC) not in sys.path:
    sys.path.insert(0, str(PATCHCORE_SRC))


patchcore = None
IMAGENET_MEAN = None
IMAGENET_STD = None
DatasetSplit = None
MVTecDataset = None
metrics = None


def lazy_imports() -> None:
    global patchcore, IMAGENET_MEAN, IMAGENET_STD, DatasetSplit, MVTecDataset, metrics
    if patchcore is not None:
        return

    import patchcore.backbones as _backbones
    import patchcore.common as _common
    import patchcore.metrics as _metrics
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
    metrics = _metrics


def now_kst_string() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, category: str, resize: int = 256, imagesize: int = 224):
        self.category = category
        self.image_paths = sorted(root.glob("*.png"))
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        return {
            "image": image,
            "mask": torch.zeros([1, *image.size()[1:]]),
            "classname": self.category,
            "anomaly": "good",
            "is_anomaly": 0,
            "image_name": image_path.name,
            "image_path": str(image_path),
        }


class MultiFolderDataset(torch.utils.data.Dataset):
    def __init__(self, roots: list[Path], category: str, resize: int = 256, imagesize: int = 224):
        self.category = category
        self.image_paths: list[Path] = []
        for root in roots:
            self.image_paths.extend(sorted(root.glob("*.png")))
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        return {
            "image": image,
            "mask": torch.zeros([1, *image.size()[1:]]),
            "classname": self.category,
            "anomaly": image_path.parent.name,
            "is_anomaly": 1,
            "image_name": image_path.name,
            "image_path": str(image_path),
        }


class ManifestSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], category: str, resize: int = 256, imagesize: int = 224):
        self.category = category
        self.entries = entries
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image_path = Path(entry["source_path"])
        with Image.open(image_path) as image_obj:
            image = image_obj.convert("RGB")
        image = apply_augmentation(
            image,
            augmentation_type=entry["augmentation_type"],
            severity=entry["severity"],
            seed=entry["seed"],
            params=entry["params"],
        )
        image = self.transform_img(image)
        return {
            "image": image,
            "mask": torch.zeros([1, *image.size()[1:]]),
            "classname": self.category,
            "anomaly": "good",
            "is_anomaly": 0,
            "image_name": entry["source_id"],
            "image_path": str(image_path),
        }


def build_patchcore(device: torch.device, imagesize: int, num_workers: int, sampler_percentage: float):
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def summarize_scores(scores: list[float], threshold: float) -> dict:
    array = np.asarray(scores, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "fpr_over_clean_max": float((array > threshold).mean()) if array.size else 0.0,
    }


def compute_image_auroc(normal_scores: list[float], anomaly_scores: list[float]) -> float:
    labels = np.asarray([0] * len(normal_scores) + [1] * len(anomaly_scores), dtype=np.int32)
    scores = np.asarray(normal_scores + anomaly_scores, dtype=np.float32)
    return float(metrics.compute_imagewise_retrieval_metrics(scores, labels)["auroc"] * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--augmentation-type")
    parser.add_argument("--augmentation-types", nargs="+")
    parser.add_argument("--input-root", default="data/query_normal_clean")
    parser.add_argument("--data-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampler-percentage", type=float, default=0.001)
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/patchcore_manifest_shift",
    )
    parser.add_argument(
        "--log-dir",
        default="experiments/validation/condition_shift_baseline/reports/patchcore_manifest_shift/logs",
    )
    args = parser.parse_args()

    lazy_imports()
    device = torch.device("cpu")
    output_dir = REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MVTecDataset(
        str(REPO_ROOT / args.data_root),
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TRAIN,
    )
    clean_dataset = ImageFolderDataset(
        REPO_ROOT / args.data_root / args.category / "test" / "good",
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )
    anomaly_dataset = MultiFolderDataset(
        [
            REPO_ROOT / args.data_root / args.category / "test" / "logical_anomalies",
            REPO_ROOT / args.data_root / args.category / "test" / "structural_anomalies",
        ],
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )
    if args.manifest:
        all_entries = [
            entry for entry in load_manifest(Path(args.manifest)) if entry["category"] == args.category
        ]
        manifest_repr = str(Path(args.manifest))
    elif args.augmentation_type or args.augmentation_types:
        augmentation_types = []
        if args.augmentation_type:
            augmentation_types.append(args.augmentation_type)
        if args.augmentation_types:
            augmentation_types.extend(args.augmentation_types)
        augmentation_types = list(dict.fromkeys(augmentation_types))
        all_entries = [
            entry
            for entry in build_manifest_entries(
                REPO_ROOT / args.input_root,
                augmentations=augmentation_types,
                severities=["low", "medium", "high"],
                seed=20260420,
            )
            if entry["category"] == args.category
        ]
        manifest_repr = f"in_memory:{','.join(augmentation_types)}"
    else:
        raise ValueError("Either --manifest or --augmentation-type(s) is required.")
    grouped: dict[str, dict[str, list[dict]]] = {}
    augmentation_types_seen: list[str] = []
    for entry in all_entries:
        aug_type = entry["augmentation_type"]
        if aug_type not in augmentation_types_seen:
            augmentation_types_seen.append(aug_type)
        grouped.setdefault(aug_type, {}).setdefault(entry["severity"], []).append(entry)

    model = build_patchcore(device, args.imagesize, args.num_workers, args.sampler_percentage)
    model.fit(make_loader(train_dataset, args.batch_size, args.num_workers))

    clean_scores, _, _, _ = model.predict(make_loader(clean_dataset, args.batch_size, args.num_workers))
    clean_scores = [float(score) for score in clean_scores]
    anomaly_scores, _, _, _ = model.predict(make_loader(anomaly_dataset, args.batch_size, args.num_workers))
    anomaly_scores = [float(score) for score in anomaly_scores]
    clean_threshold = float(np.max(clean_scores))
    results = {
        "updated_at": now_kst_string(),
        "category": args.category,
        "manifest": manifest_repr,
        "augmentation_types": augmentation_types_seen,
        "threshold_policy": "clean_max",
        "clean_good": summarize_scores(clean_scores, clean_threshold),
        "clean_anomaly": summarize_scores(anomaly_scores, clean_threshold),
        "clean_image_auroc": compute_image_auroc(clean_scores, anomaly_scores),
        "augmentations": {},
    }

    for aug_type, severity_groups in sorted(grouped.items()):
        results["augmentations"][aug_type] = {}
        for severity, entries in sorted(severity_groups.items()):
            dataset = ManifestSubsetDataset(
                entries,
                category=args.category,
                resize=args.resize,
                imagesize=args.imagesize,
            )
            scores, _, _, _ = model.predict(make_loader(dataset, args.batch_size, args.num_workers))
            scores = [float(score) for score in scores]
            summary = summarize_scores(scores, clean_threshold)
            summary["mean_score_shift"] = summary["mean"] - results["clean_good"]["mean"]
            summary["image_auroc_vs_clean_anomaly"] = compute_image_auroc(scores, anomaly_scores)
            results["augmentations"][aug_type][severity] = summary

    output_suffix = (
        augmentation_types_seen[0]
        if len(augmentation_types_seen) == 1
        else "multi"
    )
    output_path = output_dir / f"{args.category}_{output_suffix}.json"
    log_path = REPO_ROOT / args.log_dir / f"{args.category}_{output_suffix}.log.txt"
    summary = build_summary(
        baseline="PatchCore",
        dataset="mvtec_loco",
        class_name=args.category,
        eval_type="manifest_shift",
        device=str(device),
        output_path=output_path,
        log_path=log_path,
        config={
            "manifest": manifest_repr,
            "augmentation_types": augmentation_types_seen,
            "input_root": str((REPO_ROOT / args.input_root).resolve()),
            "data_root": str((REPO_ROOT / args.data_root).resolve()),
            "resize": args.resize,
            "imagesize": args.imagesize,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "sampler_percentage": args.sampler_percentage,
            "threshold_policy": "clean_max",
        },
        metrics={
            "clean_image_auroc": results["clean_image_auroc"],
            "clean_good_mean": results["clean_good"]["mean"],
            "clean_good_fpr_over_clean_max": results["clean_good"]["fpr_over_clean_max"],
            "clean_anomaly_mean": results["clean_anomaly"]["mean"],
            "clean_anomaly_fpr_over_clean_max": results["clean_anomaly"]["fpr_over_clean_max"],
        },
        paths={
            "repo_root": REPO_ROOT,
            "input_root": (REPO_ROOT / args.input_root).resolve(),
            "data_root": (REPO_ROOT / args.data_root).resolve(),
        },
        payload=results,
    )
    write_log(
        log_path,
        [
            "runner=run_patchcore_manifest_shift.py",
            "baseline=PatchCore",
            "dataset=mvtec_loco",
            f"class_name={args.category}",
            "eval_type=manifest_shift",
            f"manifest={manifest_repr}",
            f"augmentation_types={','.join(augmentation_types_seen)}",
            f"output_path={output_path}",
        ],
    )
    write_summary(summary, output_path)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
