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

from augmentation_runtime import apply_augmentation, load_manifest


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "index.md").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from: {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
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


def now_kst_string() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, category: str, resize: int = 256, imagesize: int = 224):
        self.root = root
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


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: Path, category: str, resize: int = 256, imagesize: int = 224):
        self.category = category
        self.entries = [
            entry for entry in load_manifest(manifest_path) if entry["category"] == category
        ]
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument(
        "--manifest",
        default="experiments/validation/condition_shift_baseline/manifests/query_identity.jsonl",
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampler-percentage", type=float, default=0.001)
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/patchcore_identity_repro",
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
    direct_dataset = ImageFolderDataset(
        REPO_ROOT / args.data_root / args.category / "test" / "good",
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )
    manifest_dataset = ManifestDataset(
        REPO_ROOT / args.manifest,
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )

    model = build_patchcore(device, args.imagesize, args.num_workers, args.sampler_percentage)
    model.fit(make_loader(train_dataset, args.batch_size, args.num_workers))

    direct_scores, _, _, _ = model.predict(
        make_loader(direct_dataset, args.batch_size, args.num_workers)
    )
    manifest_scores, _, _, _ = model.predict(
        make_loader(manifest_dataset, args.batch_size, args.num_workers)
    )

    direct_scores = [float(score) for score in direct_scores]
    manifest_scores = [float(score) for score in manifest_scores]
    per_sample = []
    abs_diffs = []

    for image_path, direct_score, manifest_score in zip(
        direct_dataset.image_paths, direct_scores, manifest_scores
    ):
        diff = manifest_score - direct_score
        abs_diff = abs(diff)
        abs_diffs.append(abs_diff)
        per_sample.append(
            {
                "source_id": image_path.name,
                "direct_score": direct_score,
                "manifest_identity_score": manifest_score,
                "diff": diff,
                "abs_diff": abs_diff,
            }
        )

    report = {
        "updated_at": now_kst_string(),
        "category": args.category,
        "manifest": str(REPO_ROOT / args.manifest),
        "sampler_percentage": args.sampler_percentage,
        "direct_clean": summarize(direct_scores),
        "manifest_identity": summarize(manifest_scores),
        "score_diff": {
            "mean_abs_diff": float(np.mean(abs_diffs)) if abs_diffs else 0.0,
            "max_abs_diff": float(np.max(abs_diffs)) if abs_diffs else 0.0,
            "allclose_atol_1e-6": bool(np.allclose(direct_scores, manifest_scores, atol=1e-6)),
            "allclose_atol_1e-5": bool(np.allclose(direct_scores, manifest_scores, atol=1e-5)),
        },
        "per_sample": per_sample,
    }

    output_path = output_dir / f"{args.category}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report["score_diff"], ensure_ascii=True, indent=2))
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
