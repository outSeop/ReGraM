from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.augmentation_runtime import apply_augmentation  # noqa: E402


class AugmentationRuntimeTests(unittest.TestCase):
    def test_position_shift_shrinks_and_places_without_wraparound(self) -> None:
        source = Image.new("RGB", (10, 10), color=(255, 255, 255))

        shifted = apply_augmentation(
            source,
            augmentation_type="position_shift",
            severity="high",
            seed=1,
            params={
                "scale": 0.5,
                "placement": "bottom_right",
                "fill_color": [0, 0, 0],
            },
        )

        self.assertEqual(shifted.size, source.size)
        self.assertEqual(shifted.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(shifted.getpixel((4, 4)), (0, 0, 0))
        self.assertEqual(shifted.getpixel((5, 5)), (255, 255, 255))
        self.assertEqual(shifted.getpixel((9, 9)), (255, 255, 255))

    def test_position_shift_reinterprets_legacy_max_ratio_as_shrink(self) -> None:
        source = Image.new("RGB", (10, 10), color=(255, 255, 255))

        shifted = apply_augmentation(
            source,
            augmentation_type="position_shift",
            severity="high",
            seed=1,
            params={
                "max_ratio": 0.1,
                "placement": "top_left",
                "fill_color": [0, 0, 0],
            },
        )

        self.assertEqual(shifted.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(shifted.getpixel((7, 7)), (255, 255, 255))
        self.assertEqual(shifted.getpixel((8, 8)), (0, 0, 0))
        self.assertEqual(shifted.getpixel((9, 9)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
