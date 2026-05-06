from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from relation.sam_lad_components import (  # noqa: E402
    SamLadComponentConfig,
    SamLadComponentModel,
    extract_sam_lad_components_from_masks,
)


def mask_with_box(box: tuple[int, int, int, int], *, shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class FakeMaskGenerator:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.masks = masks

    def generate(self, _image: np.ndarray) -> list[dict[str, np.ndarray]]:
        return [{"segmentation": mask} for mask in self.masks]


class SamLadComponentTests(unittest.TestCase):
    def test_sam_lad_merges_tiny_masks_into_one_component(self) -> None:
        raw_masks = [
            {"segmentation": mask_with_box((10, 10, 30, 30))},
            {"segmentation": mask_with_box((60, 60, 80, 80))},
            {"segmentation": mask_with_box((42, 42, 45, 45))},
            {"segmentation": mask_with_box((47, 42, 50, 45))},
        ]

        result = extract_sam_lad_components_from_masks(
            raw_masks,
            image_size=(100, 100),
            config=SamLadComponentConfig(
                min_area_ratio=0.0005,
                small_area_ratio=0.006,
                max_area_ratio=0.85,
            ),
        )

        self.assertEqual(result.raw_mask_count, 4)
        self.assertEqual(result.merged_small_count, 2)
        self.assertEqual(len(result.components), 3)
        self.assertIn("sam_lad_merged_small", {component.source for component in result.components})

    def test_sam_lad_drops_background_sized_masks(self) -> None:
        result = extract_sam_lad_components_from_masks(
            [
                {"segmentation": mask_with_box((0, 0, 100, 100))},
                {"segmentation": mask_with_box((15, 15, 35, 35))},
            ],
            image_size=(100, 100),
            config=SamLadComponentConfig(max_area_ratio=0.5),
        )

        self.assertEqual(len(result.components), 1)
        self.assertEqual(result.components[0].area, 400)
        self.assertIn("dropped_large_masks=1", result.note)

    def test_sam_lad_component_model_accepts_injected_mask_generator(self) -> None:
        model = SamLadComponentModel(
            mask_generator=FakeMaskGenerator(
                [
                    mask_with_box((10, 10, 30, 30)),
                    mask_with_box((60, 60, 80, 80)),
                ]
            ),
            config=SamLadComponentConfig(),
        )

        result = model.extract(Image.new("RGB", (100, 100), "white"))

        self.assertEqual(result.component_source, "sam_lad")
        self.assertEqual(len(result.components), 2)


if __name__ == "__main__":
    unittest.main()
