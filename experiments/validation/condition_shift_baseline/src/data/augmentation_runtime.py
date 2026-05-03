from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


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

_IDENTITY_PARAMS = {"identity": {"none": {}}}

# Hardcoded fallback used only when PyYAML is unavailable at import time.
# When PyYAML is present, augmentation_protocol.yaml is the source of truth and
# this fallback is cross-checked against it — any drift raises AssertionError.
_FALLBACK_DEFAULT_PARAMS = {
    **_IDENTITY_PARAMS,
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
        "low": {
            "center_shift_ratio": 0.03,
            "placement": "seeded_corner",
            "fill_mode": "border_median",
        },
        "medium": {
            "center_shift_ratio": 0.06,
            "placement": "seeded_corner",
            "fill_mode": "border_median",
        },
        "high": {
            "center_shift_ratio": 0.10,
            "placement": "seeded_corner",
            "fill_mode": "border_median",
        },
    },
    "low_light": {
        "low": {"brightness": 0.75, "contrast": 0.85},
        "medium": {"brightness": 0.55, "contrast": 0.75},
        "high": {"brightness": 0.40, "contrast": 0.65},
    },
}


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "index.md").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from: {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
AUGMENTATION_PROTOCOL_PATH = EXPERIMENT_DIR / "configs" / "augmentation_protocol.yaml"


def load_augmentation_protocol(
    path: Path | None = None,
) -> dict[str, dict[str, dict]] | None:
    try:
        import yaml  # type: ignore
    except ImportError:
        return None

    target = path or AUGMENTATION_PROTOCOL_PATH
    doc = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    params: dict[str, dict[str, dict]] = {}
    for entry in doc.get("augmentations") or []:
        if not entry.get("enabled", True):
            continue
        params[entry["name"]] = dict(entry["params"])
    return params


def _resolve_default_params() -> dict[str, dict[str, dict]]:
    yaml_params = load_augmentation_protocol()
    if yaml_params is None:
        return dict(_FALLBACK_DEFAULT_PARAMS)

    fallback_non_identity = {
        name: severities
        for name, severities in _FALLBACK_DEFAULT_PARAMS.items()
        if name != "identity"
    }
    assert yaml_params == fallback_non_identity, (
        "augmentation_protocol.yaml drift vs. hardcoded fallback. "
        "YAML is the source of truth; update _FALLBACK_DEFAULT_PARAMS to match. "
        f"yaml_keys={sorted(yaml_params)} fallback_keys={sorted(fallback_non_identity)}"
    )
    return {**_IDENTITY_PARAMS, **yaml_params}


DEFAULT_PARAMS = _resolve_default_params()


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
    input_root = input_root.resolve()
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
                            "source_path": str(image_path.resolve().relative_to(REPO_ROOT)),
                            "source_path_mode": "repo_relative",
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


_POSITION_CORNER_PLACEMENTS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _position_shift_scale(params: dict) -> float:
    if "scale" in params:
        scale = float(params["scale"])
    else:
        # Backward compatibility: older manifests stored max_ratio for the
        # previous cyclic offset implementation. Reinterpret it as the desired
        # center displacement, then shrink enough to place the full image
        # without wraparound or cropping.
        center_shift_ratio = float(
            params.get("center_shift_ratio", params.get("max_ratio", 0.0))
        )
        scale = 1.0 - (2.0 * center_shift_ratio)
    return min(1.0, max(0.05, scale))


def _resolve_position_shift_placement(params: dict, seed: int) -> str:
    placement = params.get("placement", params.get("anchor", "seeded_corner"))
    if isinstance(placement, list):
        candidates = tuple(str(item) for item in placement)
        if not candidates:
            raise ValueError("position_shift placement list cannot be empty")
        return random.Random(seed).choice(candidates)
    if placement in {"seeded_corner", "random_corner"}:
        return random.Random(seed).choice(_POSITION_CORNER_PLACEMENTS)
    return str(placement)


def _resolve_position_shift_fill(image: Image.Image, params: dict) -> tuple[int, int, int]:
    fill_color = params.get("fill_color")
    if fill_color is not None:
        if isinstance(fill_color, str):
            named = {
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "gray": (127, 127, 127),
                "grey": (127, 127, 127),
            }
            if fill_color not in named:
                raise ValueError(f"Unsupported position_shift fill_color: {fill_color}")
            return named[fill_color]
        if len(fill_color) != 3:
            raise ValueError("position_shift fill_color must have three channels")
        return tuple(int(value) for value in fill_color)

    fill_mode = params.get("fill_mode", "border_median")
    if fill_mode == "black":
        return (0, 0, 0)
    if fill_mode == "white":
        return (255, 255, 255)
    if fill_mode != "border_median":
        raise ValueError(f"Unsupported position_shift fill_mode: {fill_mode}")

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    border = np.concatenate(
        [
            array[0, :, :],
            array[-1, :, :],
            array[:, 0, :],
            array[:, -1, :],
        ],
        axis=0,
    )
    return tuple(int(value) for value in np.median(border, axis=0))


def _position_shift_offset(
    image_size: tuple[int, int],
    shifted_size: tuple[int, int],
    placement: str,
) -> tuple[int, int]:
    width, height = image_size
    shifted_width, shifted_height = shifted_size
    max_x = width - shifted_width
    max_y = height - shifted_height
    offsets = {
        "top_left": (0, 0),
        "top_right": (max_x, 0),
        "bottom_left": (0, max_y),
        "bottom_right": (max_x, max_y),
        "center": (max_x // 2, max_y // 2),
    }
    if placement not in offsets:
        raise ValueError(
            "Unsupported position_shift placement: "
            f"{placement}. Expected one of {sorted(offsets)} or seeded_corner."
        )
    return offsets[placement]


def _apply_position_shift(image: Image.Image, *, seed: int, params: dict) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = _position_shift_scale(params)
    shifted_size = (
        max(1, min(width, int(round(width * scale)))),
        max(1, min(height, int(round(height * scale)))),
    )
    placement = _resolve_position_shift_placement(params, seed)
    offset = _position_shift_offset(image.size, shifted_size, placement)
    fill = _resolve_position_shift_fill(image, params)

    shifted = image.resize(shifted_size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", image.size, fill)
    canvas.paste(shifted, offset)
    return canvas


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
        return _apply_position_shift(image, seed=seed, params=params)

    if augmentation_type == "low_light":
        output = ImageEnhance.Brightness(image).enhance(params["brightness"])
        return ImageEnhance.Contrast(output).enhance(params["contrast"])

    raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
