"""Preview panel generation for wandb image logging."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from augmentation_runtime import apply_augmentation
from manifest_paths import resolve_manifest_image_path


def build_preview_panel(entry: dict, image_path: Path) -> Image.Image:
    with Image.open(image_path) as image_obj:
        original = image_obj.convert("RGB")
    augmented = apply_augmentation(
        original.copy(),
        augmentation_type=entry["augmentation_type"],
        severity=entry["severity"],
        seed=entry["seed"],
        params=entry["params"],
    )
    original_thumb = original.resize((224, 224))
    augmented_thumb = augmented.resize((224, 224))
    canvas = Image.new("RGB", (224 * 2, 224), color=(255, 255, 255))
    canvas.paste(original_thumb, (0, 0))
    canvas.paste(augmented_thumb, (224, 0))
    return canvas


def build_preview_images(
    entries: list[dict], *, max_images: int
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for entry in entries[:max_images]:
        image_path = resolve_manifest_image_path(entry)
        panel = build_preview_panel(entry, image_path)
        previews.append(
            {
                "image": panel,
                "caption": (
                    f"{entry['augmentation_type']} | {entry['severity']} | "
                    f"{entry['source_id']}"
                ),
            }
        )
    return previews
