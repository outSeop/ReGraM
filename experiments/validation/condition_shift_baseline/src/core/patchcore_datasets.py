"""Dataset classes shared across PatchCore runners."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from augmentation_runtime import apply_augmentation, load_manifest
from patchcore_factory import get_patchcore
from repo_paths import REPO_ROOT


def _build_transform(resize: int, imagesize: int):
    ns = get_patchcore()
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=ns.IMAGENET_MEAN, std=ns.IMAGENET_STD),
        ]
    )


def resolve_manifest_image_path(entry: dict) -> Path:
    source_path = Path(entry["source_path"])
    path_mode = entry.get("source_path_mode", "absolute")
    if source_path.is_absolute():
        return source_path
    if path_mode == "repo_relative":
        return REPO_ROOT / source_path
    return source_path.resolve()


def load_category_entries(manifest_path: Path, category: str) -> list[dict]:
    return [
        entry for entry in load_manifest(manifest_path) if entry["category"] == category
    ]


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: Path, category: str, resize: int = 256, imagesize: int = 224
    ):
        self.category = category
        self.image_paths = sorted(root.glob("*.png"))
        self.transform_img = _build_transform(resize, imagesize)

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
    def __init__(
        self,
        roots: list[Path],
        category: str,
        resize: int = 256,
        imagesize: int = 224,
    ):
        self.category = category
        self.image_paths: list[Path] = []
        for root in roots:
            self.image_paths.extend(sorted(root.glob("*.png")))
        self.transform_img = _build_transform(resize, imagesize)

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
    def __init__(
        self,
        entries: list[dict],
        category: str,
        resize: int = 256,
        imagesize: int = 224,
    ):
        self.category = category
        self.entries = entries
        self.transform_img = _build_transform(resize, imagesize)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image_path = resolve_manifest_image_path(entry)
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
