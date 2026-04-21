from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter


DEFAULT_SEVERITIES = ("low", "medium", "high")
DEFAULT_AUGMENTATIONS = (
    "brightness",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "compression",
    "low_resolution",
    "position_shift",
    "low_light",
)

DEFAULT_PARAMS = {
    "identity": {
        "none": {},
    },
    "brightness": {
        "low": {"factor": 0.80},
        "medium": {"factor": 0.60},
        "high": {"factor": 0.40},
    },
    "gaussian_blur": {
        "low": {"radius": 2.0},
        "medium": {"radius": 4.0},
        "high": {"radius": 6.0},
    },
    "motion_blur": {
        "low": {"kernel_size": 5},
        "medium": {"kernel_size": 9},
        "high": {"kernel_size": 13},
    },
    "gaussian_noise": {
        "low": {"sigma": 24.0},
        "medium": {"sigma": 48.0},
        "high": {"sigma": 72.0},
    },
    "compression": {
        "low": {"quality": 45},
        "medium": {"quality": 20},
        "high": {"quality": 10},
    },
    "low_resolution": {
        "low": {"scale": 0.75},
        "medium": {"scale": 0.50},
        "high": {"scale": 0.33},
    },
    "position_shift": {
        "low": {"max_ratio": 0.03},
        "medium": {"max_ratio": 0.06},
        "high": {"max_ratio": 0.10},
    },
    "low_light": {
        "low": {"brightness": 0.75, "contrast": 0.85},
        "medium": {"brightness": 0.55, "contrast": 0.75},
        "high": {"brightness": 0.40, "contrast": 0.65},
    },
}


def load_manifest(manifest_path: Path) -> list[dict]:
    entries: list[dict] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def build_manifest_entries(
    input_root: Path,
    *,
    augmentations: list[str] | None = None,
    severities: list[str] | None = None,
    identity_only: bool = False,
    seed: int = 20260420,
) -> list[dict]:
    entries: list[dict] = []
    if identity_only:
        augmentations = ["identity"]
        severities = ["none"]
    else:
        augmentations = augmentations or list(DEFAULT_AUGMENTATIONS)
        severities = severities or list(DEFAULT_SEVERITIES)

    for category_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        category = category_dir.name
        for image_path in sorted(category_dir.glob("*.png")):
            source_id = image_path.name
            base_seed = seed + sum(ord(ch) for ch in f"{category}/{source_id}")
            for augmentation_type in augmentations:
                for severity in severities:
                    params = DEFAULT_PARAMS[augmentation_type][severity]
                    item_seed = base_seed + sum(
                        ord(ch) for ch in f"{augmentation_type}/{severity}"
                    )
                    entries.append(
                        {
                            "source_path": str(image_path.resolve()),
                            "category": category,
                            "source_id": source_id,
                            "augmentation_type": augmentation_type,
                            "severity": severity,
                            "seed": item_seed,
                            "params": params,
                        }
                    )
    return entries


def write_manifest(entries: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def _from_numpy(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def _motion_kernel(size: int) -> list[float]:
    kernel = [0.0] * (size * size)
    center_row = size // 2
    for col in range(size):
        kernel[center_row * size + col] = 1.0 / size
    return kernel


def _ensure_odd_kernel_size(size: int) -> int:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return size


def _apply_horizontal_motion_blur(image: Image.Image, size: int) -> Image.Image:
    size = _ensure_odd_kernel_size(size)
    half = size // 2
    array = _to_numpy(image)
    shifted_sum = np.zeros_like(array, dtype=np.float32)
    for offset in range(-half, half + 1):
        shifted_sum += np.roll(array, shift=offset, axis=1)
    return _from_numpy(shifted_sum / size)


def apply_augmentation(
    image: Image.Image,
    *,
    augmentation_type: str,
    severity: str,
    seed: int,
    params: dict | None = None,
) -> Image.Image:
    params = params or DEFAULT_PARAMS[augmentation_type][severity]

    if augmentation_type == "identity":
        return image.copy()

    if augmentation_type == "brightness":
        return ImageEnhance.Brightness(image).enhance(params["factor"])

    if augmentation_type == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=params["radius"]))

    if augmentation_type == "motion_blur":
        size = _ensure_odd_kernel_size(params["kernel_size"])
        return _apply_horizontal_motion_blur(image, size)

    if augmentation_type == "gaussian_noise":
        rng = np.random.default_rng(seed)
        array = _to_numpy(image)
        noise = rng.normal(0.0, params["sigma"], size=array.shape)
        return _from_numpy(array + noise)

    if augmentation_type == "compression":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=params["quality"], optimize=False)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if augmentation_type == "low_resolution":
        width, height = image.size
        scale = params["scale"]
        downsampled = image.resize(
            (max(8, int(width * scale)), max(8, int(height * scale))),
            resample=Image.Resampling.BILINEAR,
        )
        return downsampled.resize((width, height), resample=Image.Resampling.BICUBIC)

    if augmentation_type == "position_shift":
        rng = random.Random(seed)
        width, height = image.size
        max_dx = int(width * params["max_ratio"])
        max_dy = int(height * params["max_ratio"])
        dx = rng.randint(-max_dx, max_dx)
        dy = rng.randint(-max_dy, max_dy)
        return ImageChops.offset(image, dx, dy)

    if augmentation_type == "low_light":
        output = ImageEnhance.Brightness(image).enhance(params["brightness"])
        return ImageEnhance.Contrast(output).enhance(params["contrast"])

    raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
