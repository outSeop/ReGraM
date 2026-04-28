"""Dataset classes shared across PatchCore runners."""

from __future__ import annotations

import sys
from pathlib import Path

_cursor = Path(__file__).resolve()
while _cursor.name != "src" and _cursor.parent != _cursor:
    _cursor = _cursor.parent
_SRC_ROOT = _cursor
_CORE_ROOT = _SRC_ROOT / "core"
for _p in (_SRC_ROOT, _CORE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from augmentation_runtime import apply_augmentation, load_manifest
from manifest_paths import resolve_manifest_image_path


def _build_transform(resize: int, imagesize: int):
    from patchcore_factory import get_patchcore  # noqa: WPS433

    ns = get_patchcore()
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=ns.IMAGENET_MEAN, std=ns.IMAGENET_STD),
        ]
    )


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
