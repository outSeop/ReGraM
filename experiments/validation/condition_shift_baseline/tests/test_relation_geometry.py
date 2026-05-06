from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from relation.geometry import (  # noqa: E402
    apply_position_transform,
    relation_score_bundle,
    resolve_position_shift_transform,
)


class RelationGeometryTests(unittest.TestCase):
    def test_position_shift_transform_supports_legacy_max_ratio(self) -> None:
        transform = resolve_position_shift_transform(
            (100, 80),
            {"max_ratio": 0.1, "placement": "top_right"},
            seed=123,
        )

        self.assertAlmostEqual(transform.scale, 0.8)
        self.assertEqual(transform.shifted_size, (80, 64))
        self.assertEqual(transform.offset, (20.0, 0.0))

    def test_normalized_relation_scores_ignore_scale_and_translation(self) -> None:
        reference = np.asarray(
            [
                [10.0, 10.0],
                [40.0, 10.0],
                [10.0, 50.0],
                [35.0, 45.0],
            ]
        )
        transform = resolve_position_shift_transform(
            (100, 100),
            {"scale": 0.75, "placement": "bottom_right"},
            seed=123,
        )
        query = apply_position_transform(reference, transform)
        scores = relation_score_bundle(reference, query)

        self.assertGreater(scores["s_abs"], 0.0)
        self.assertGreater(scores["s_centered_raw"], 0.0)
        self.assertGreater(scores["s_pair_raw"], 0.0)
        self.assertAlmostEqual(scores["s_centered_norm"], 0.0, places=12)
        self.assertAlmostEqual(scores["s_pair_norm"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
