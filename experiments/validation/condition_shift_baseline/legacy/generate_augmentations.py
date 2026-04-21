from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter


BRIGHTNESS_FACTORS = {"low": 0.80, "medium": 0.60, "high": 0.40}
GAUSSIAN_BLUR_RADII = {"low": 2.0, "medium": 4.0, "high": 6.0}
MOTION_KERNEL_SIZES = {"low": 5, "medium": 9, "high": 13}

GAUSSIAN_NOISE_SIGMA = {"low": 8.0, "medium": 16.0, "high": 24.0}

JPEG_QUALITY = {"low": 45, "medium": 20, "high": 10}
LOW_RES_SCALE = {"low": 0.75, "medium": 0.50, "high": 0.33}

POSITION_SHIFT_RATIO = {"low": 0.03, "medium": 0.06, "high": 0.10}
LOW_LIGHT_BRIGHTNESS = {"low": 0.75, "medium": 0.55, "high": 0.40}
LOW_LIGHT_CONTRAST = {"low": 0.85, "medium": 0.75, "high": 0.65}


def _to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def _from_numpy(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def apply_brightness(image: Image.Image, severity: str) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(BRIGHTNESS_FACTORS[severity])


def apply_gaussian_blur(image: Image.Image, severity: str) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_BLUR_RADII[severity]))


def _motion_kernel(size: int) -> list[float]:
    kernel = [0.0] * (size * size)
    center_row = size // 2
    for col in range(size):
        kernel[center_row * size + col] = 1.0 / size
    return kernel


def apply_motion_blur(image: Image.Image, severity: str) -> Image.Image:
    size = MOTION_KERNEL_SIZES[severity]
    kernel = _motion_kernel(size)
    return image.filter(ImageFilter.Kernel((size, size), kernel, scale=sum(kernel)))


def apply_gaussian_noise(image: Image.Image, severity: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = _to_numpy(image)
    noise = rng.normal(0.0, GAUSSIAN_NOISE_SIGMA[severity], size=array.shape)
    return _from_numpy(array + noise)


def apply_compression(image: Image.Image, severity: str) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY[severity], optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_low_resolution(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    scale = LOW_RES_SCALE[severity]
    downsampled_size = (max(8, int(width * scale)), max(8, int(height * scale)))
    reduced = image.resize(downsampled_size, resample=Image.Resampling.BILINEAR)
    return reduced.resize((width, height), resample=Image.Resampling.BICUBIC)


def apply_position_shift(image: Image.Image, severity: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    width, height = image.size
    max_dx = int(width * POSITION_SHIFT_RATIO[severity])
    max_dy = int(height * POSITION_SHIFT_RATIO[severity])
    dx = rng.randint(-max_dx, max_dx)
    dy = rng.randint(-max_dy, max_dy)
    return ImageChops.offset(image, dx, dy)


def apply_low_light(image: Image.Image, severity: str) -> Image.Image:
    output = ImageEnhance.Brightness(image).enhance(LOW_LIGHT_BRIGHTNESS[severity])
    output = ImageEnhance.Contrast(output).enhance(LOW_LIGHT_CONTRAST[severity])
    return output


def apply_augmentation(image: Image.Image, aug_type: str, severity: str, seed: int) -> Image.Image:
    if aug_type == "brightness":
        return apply_brightness(image, severity)
    if aug_type == "gaussian_blur":
        return apply_gaussian_blur(image, severity)
    if aug_type == "motion_blur":
        return apply_motion_blur(image, severity)
    if aug_type == "gaussian_noise":
        return apply_gaussian_noise(image, severity, seed)
    if aug_type == "compression":
        return apply_compression(image, severity)
    if aug_type == "low_resolution":
        return apply_low_resolution(image, severity)
    if aug_type == "position_shift":
        return apply_position_shift(image, severity, seed)
    if aug_type == "low_light":
        return apply_low_light(image, severity)
    raise ValueError(f"Unsupported augmentation type: {aug_type}")


def build_tasks(
    input_root: Path,
    output_root: Path,
    augmentations: list[str],
    severities: list[str],
    seed: int,
) -> list[dict]:
    tasks: list[dict] = []
    for category_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        category = category_dir.name
        for image_path in sorted(category_dir.glob("*.png")):
            source_id = image_path.name
            base_seed = seed + sum(ord(ch) for ch in f"{category}/{source_id}")
            for aug_type in augmentations:
                for severity in severities:
                    aug_seed = base_seed + sum(ord(ch) for ch in f"{aug_type}/{severity}")
                    tasks.append(
                        {
                            "source_id": source_id,
                            "category": category,
                            "augmentation_type": aug_type,
                            "severity": severity,
                            "seed": aug_seed,
                            "input_path": str(image_path),
                            "output_path": str(output_root / aug_type / severity / category / source_id),
                        }
                    )
    return tasks


def run_task(task: dict) -> dict:
    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image_obj:
        image = image_obj.convert("RGB")
        augmented = apply_augmentation(image, task["augmentation_type"], task["severity"], task["seed"])
        augmented.save(output_path, compress_level=1)
    return task


def write_progress_files(
    output_root: Path,
    total: int,
    completed: int,
    started_at: float,
    by_aug: Counter,
    by_category: Counter,
    workers: int,
) -> None:
    elapsed = time.time() - started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(total - completed, 0)
    eta_seconds = int(remaining / rate) if rate > 0 else None
    percent = round((completed / total) * 100, 2) if total else 0.0

    progress_json = {
        "total": total,
        "completed": completed,
        "remaining": remaining,
        "percent": percent,
        "elapsed_seconds": round(elapsed, 2),
        "items_per_second": round(rate, 3),
        "eta_seconds": eta_seconds,
        "workers": workers,
        "by_augmentation": dict(sorted(by_aug.items())),
        "by_category": dict(sorted(by_category.items())),
    }
    (output_root / "progress.json").write_text(
        json.dumps(progress_json, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Augmentation Progress",
        "",
        f"- completed: `{completed} / {total}`",
        f"- percent: `{percent}%`",
        f"- elapsed: `{round(elapsed, 1)} sec`",
        f"- speed: `{round(rate, 2)} items/sec`",
        f"- eta_seconds: `{eta_seconds}`",
        f"- workers: `{workers}`",
        "",
        "## By Augmentation",
        "",
    ]
    for key, value in sorted(by_aug.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## By Category", ""])
    for key, value in sorted(by_category.items()):
        lines.append(f"- `{key}`: `{value}`")
    (output_root / "PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmented query-only samples in parallel.")
    parser.add_argument("--input-root", default="data/query_normal_clean")
    parser.add_argument("--output-root", default="data/query_normal_augmented")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--refresh-every", type=int, default=25)
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=[
            "brightness",
            "gaussian_blur",
            "motion_blur",
            "gaussian_noise",
            "compression",
            "low_resolution",
            "position_shift",
            "low_light",
        ],
    )
    parser.add_argument("--severities", nargs="+", default=["low", "medium", "high"])
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(input_root, output_root, args.augmentations, args.severities, args.seed)
    total = len(tasks)
    started_at = time.time()
    completed = 0
    by_aug: Counter = Counter()
    by_category: Counter = Counter()

    manifest_path = output_root / "manifest.jsonl"
    write_progress_files(output_root, total, completed, started_at, by_aug, by_category, args.workers)
    print(f"queued={total} workers={args.workers}", flush=True)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            for future in cf.as_completed(futures):
                task = future.result()
                manifest.write(json.dumps(task, ensure_ascii=True) + "\n")
                completed += 1
                by_aug[task["augmentation_type"]] += 1
                by_category[task["category"]] += 1

                if completed % args.refresh_every == 0 or completed == total:
                    manifest.flush()
                    write_progress_files(
                        output_root,
                        total,
                        completed,
                        started_at,
                        by_aug,
                        by_category,
                        args.workers,
                    )
                    print(f"progress={completed}/{total}", flush=True)

    write_progress_files(output_root, total, completed, started_at, by_aug, by_category, args.workers)
    print(f"done={completed}/{total}", flush=True)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
