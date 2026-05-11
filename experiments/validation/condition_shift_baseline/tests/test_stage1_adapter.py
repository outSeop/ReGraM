from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stage1_adapter import (  # noqa: E402
    build_component_prototypes,
    describe_candidate_masks,
    score_candidate_against_prototypes,
    summarize_adapter_scores,
)


def box_mask(box: tuple[int, int, int, int], *, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class Stage1AdapterTests(unittest.TestCase):
    def test_describes_candidate_masks_with_inside_ring_and_neighbor_stats(self) -> None:
        feature_map = np.ones((8, 8, 4), dtype=np.float32)
        feature_map[:, :, 0] = np.linspace(0, 1, 8)[None, :]
        raw_masks = [
            {"mask_id": "object", "mask": box_mask((8, 8, 24, 24))},
            {"mask_id": "neighbor", "mask": box_mask((22, 8, 32, 24))},
        ]

        descriptors = describe_candidate_masks(
            image_id="000.png",
            feature_map=feature_map,
            raw_masks=raw_masks,
            image_shape=(64, 64),
            source="test",
        )

        self.assertEqual(len(descriptors), 2)
        self.assertEqual(descriptors[0].candidate_id, "object")
        self.assertGreater(descriptors[0].area_ratio, 0)
        self.assertGreater(descriptors[0].neighbor_count, 0)
        self.assertGreaterEqual(descriptors[0].inside_feature_coherence, 0)

    def test_builds_prototypes_and_scores_invalid_supermask(self) -> None:
        feature_map = np.ones((8, 8, 4), dtype=np.float32)
        clean_masks = [
            {"mask_id": "a", "mask": box_mask((8, 8, 24, 24))},
            {"mask_id": "b", "mask": box_mask((36, 8, 52, 24))},
        ]
        clean_descriptors = describe_candidate_masks(
            image_id="000.png",
            feature_map=feature_map,
            raw_masks=clean_masks,
            image_shape=(64, 64),
            source="clean",
        )
        prototypes = build_component_prototypes(clean_descriptors, max_prototypes=2)

        query_descriptors = describe_candidate_masks(
            image_id="query.png",
            feature_map=feature_map,
            raw_masks=[
                {"mask_id": "super", "mask": box_mask((0, 0, 64, 64))},
                *clean_masks,
            ],
            image_shape=(64, 64),
            source="query",
        )
        supermask = next(item for item in query_descriptors if item.candidate_id == "super")
        score = score_candidate_against_prototypes(supermask, prototypes, too_large_area_ratio=0.55)

        self.assertEqual(score["node_type"], "invalid_supermask")
        self.assertLess(score["reliability"], 0.2)
        self.assertIn("too_large_container_or_supermask", score["debug"]["invalid_reasons"])
        summary = summarize_adapter_scores([score])
        self.assertEqual(summary["num_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
