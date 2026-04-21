from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from augmentation_runtime import DEFAULT_AUGMENTATIONS, DEFAULT_SEVERITIES, apply_augmentation


def add_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill="black")
    return canvas


def make_grid(images: list[Image.Image], columns: int) -> Image.Image:
    rows = (len(images) + columns - 1) // columns
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    grid = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")

    for index, image in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        grid.paste(image, (x, y))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Render representative augmentation samples.")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=["gaussian_noise"],
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        default=list(DEFAULT_SEVERITIES),
    )
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/sample_panels/gaussian_noise_breakfast_box_000.png",
    )
    parser.add_argument("--columns", type=int, default=2)
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image_obj:
        source = image_obj.convert("RGB")

    labeled_images = [add_label(source, "clean")]
    base_seed = 20260420 + sum(ord(ch) for ch in str(source_path))

    for augmentation_type in args.augmentations:
        if augmentation_type not in DEFAULT_AUGMENTATIONS:
            raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
        for severity in args.severities:
            seed = base_seed + sum(ord(ch) for ch in f"{augmentation_type}/{severity}")
            augmented = apply_augmentation(
                source,
                augmentation_type=augmentation_type,
                severity=severity,
                seed=seed,
            )
            labeled_images.append(add_label(augmented, f"{augmentation_type}:{severity}"))

    grid = make_grid(labeled_images, columns=args.columns)
    grid.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
