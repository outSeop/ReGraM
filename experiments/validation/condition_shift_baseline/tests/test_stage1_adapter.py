from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stage1_adapter import (  # noqa: E402
    CandidateMaskNormalizationConfig,
    PatchGraphConfig,
    build_component_prototypes,
    describe_candidate_masks,
    normalize_candidate_masks,
    run_masked_patch_prototype_probe,
    score_candidate_against_prototypes,
    select_reliable_candidate_ids,
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
        self.assertIn("prototype_margin", score)
        self.assertIn("top2_prototype_similarity", score)

    def test_masked_patch_prototype_probe_uses_reliable_membership(self) -> None:
        feature_map = np.zeros((4, 4, 3), dtype=np.float32)
        feature_map[:, :2, 0] = 1.0
        feature_map[:, 2:, 1] = 1.0
        clean_masks = [
            {"mask_id": "left", "mask": box_mask((0, 0, 32, 64))},
            {"mask_id": "right", "mask": box_mask((32, 0, 64, 64))},
        ]
        clean_descriptors = describe_candidate_masks(
            image_id="000.png",
            feature_map=feature_map,
            raw_masks=clean_masks,
            image_shape=(64, 64),
            source="clean",
        )
        prototypes = build_component_prototypes(clean_descriptors, max_prototypes=2)
        candidate_scores = [
            {
                "candidate_id": "left",
                "node_type": "valid_component",
                "reliability": 0.9,
                "area_ratio": 0.5,
            },
            {
                "candidate_id": "right",
                "node_type": "valid_component",
                "reliability": 0.9,
                "area_ratio": 0.5,
            },
        ]

        result = run_masked_patch_prototype_probe(
            feature_map=feature_map,
            raw_masks=clean_masks,
            candidate_scores=candidate_scores,
            prototypes=prototypes,
            image_shape=(64, 64),
            config=PatchGraphConfig(min_reliability=0.5, max_area_ratio=0.6),
        )

        self.assertEqual(result.summary["num_reliable_patches"], 16)
        self.assertGreater(result.summary["edge_summary"]["num_same_mask_edges"], 0)
        self.assertGreater(result.summary["edge_summary"]["num_cross_boundary_edges"], 0)
        self.assertIn("boundary_patch_match_ratio", result.summary)
        self.assertIn("same_mask_edge_consistency", result.summary)
        self.assertIn("cross_boundary_contrast", result.summary)
        self.assertIn("teacher_quality", result.summary)
        self.assertIn("dino_teacher_quality", result.summary)
        self.assertEqual(result.membership.shape, (4, 4))

    def test_mask_smoothed_teacher_changes_only_inside_membership(self) -> None:
        feature_map = np.zeros((4, 4, 3), dtype=np.float32)
        feature_map[:, :2, 0] = 1.0
        feature_map[:, 2:, 1] = 1.0
        clean_masks = [
            {"mask_id": "left", "mask": box_mask((0, 0, 32, 64))},
            {"mask_id": "right", "mask": box_mask((32, 0, 64, 64))},
        ]
        clean_descriptors = describe_candidate_masks(
            image_id="000.png",
            feature_map=feature_map,
            raw_masks=clean_masks,
            image_shape=(64, 64),
            source="clean",
        )
        prototypes = build_component_prototypes(clean_descriptors, max_prototypes=2)
        candidate_scores = [
            {"candidate_id": "left", "node_type": "valid_component", "reliability": 0.9, "area_ratio": 0.5},
            {"candidate_id": "right", "node_type": "valid_component", "reliability": 0.9, "area_ratio": 0.5},
        ]

        result = run_masked_patch_prototype_probe(
            feature_map=feature_map,
            raw_masks=clean_masks,
            candidate_scores=candidate_scores,
            prototypes=prototypes,
            image_shape=(64, 64),
            config=PatchGraphConfig(
                target_type="mask_smoothed",
                teacher_smoothing_lambda=0.5,
                max_area_ratio=0.6,
            ),
        )

        self.assertEqual(result.teacher_probabilities.shape, result.dino_teacher_probabilities.shape)
        self.assertGreaterEqual(float(result.teacher_delta_map.mean()), 0.0)
        self.assertEqual(result.summary["config"]["target_type"], "mask_smoothed")
        self.assertIn("teacher_argmax_change_rate", result.summary)
        self.assertIn("boundary_delta_mean", result.summary)

    def test_local_and_boundary_weighted_smoothing_modes_run(self) -> None:
        probabilities = np.zeros((3, 3, 2), dtype=np.float32)
        probabilities[:, :, 0] = 0.8
        probabilities[:, :, 1] = 0.2
        probabilities[1, 1] = [0.2, 0.8]
        membership = np.zeros((3, 3), dtype=np.int32)

        from stage1_adapter import smooth_teacher_with_membership  # noqa: PLC0415

        local = smooth_teacher_with_membership(
            probabilities,
            membership,
            smoothing_mode="local_same_mask",
            smoothing_lambda=0.5,
        )
        boundary = smooth_teacher_with_membership(
            probabilities,
            membership,
            smoothing_mode="boundary_weighted_local",
            smoothing_lambda=0.3,
            inner_lambda=0.1,
            boundary_lambda=0.3,
        )

        self.assertEqual(local.shape, probabilities.shape)
        self.assertEqual(boundary.shape, probabilities.shape)
        self.assertTrue(np.allclose(local.sum(axis=-1), 1.0))
        self.assertTrue(np.allclose(boundary.sum(axis=-1), 1.0))

    def test_top_k_selection_keeps_best_after_hard_filter(self) -> None:
        scores = [
            {
                "candidate_id": "huge",
                "node_type": "low_reliability_candidate",
                "reliability": 10.0,
                "area_ratio": 0.9,
                "max_prototype_similarity": 1.0,
            },
            {
                "candidate_id": "best",
                "node_type": "low_reliability_candidate",
                "reliability": 0.2,
                "area_ratio": 0.1,
                "max_prototype_similarity": 0.9,
            },
            {
                "candidate_id": "second",
                "node_type": "small_detail",
                "reliability": 0.01,
                "area_ratio": 0.002,
                "max_prototype_similarity": 0.7,
            },
        ]

        selected, debug = select_reliable_candidate_ids(
            scores,
            config=PatchGraphConfig(reliable_selection_mode="top_k", top_k_reliable_masks=2, max_area_ratio=0.3),
        )

        self.assertNotIn("huge", selected)
        self.assertEqual(selected, {"best", "second"})
        self.assertEqual(debug["num_selected"], 2)

        none_selected, none_debug = select_reliable_candidate_ids(
            scores,
            config=PatchGraphConfig(reliable_selection_mode="none"),
        )
        self.assertEqual(none_selected, set())
        self.assertEqual(none_debug["num_selected"], 0)

    def test_normalizes_large_and_small_candidate_masks(self) -> None:
        raw_masks = [
            {"mask_id": "huge", "mask": box_mask((0, 0, 64, 64))},
            {"mask_id": "medium", "mask": box_mask((5, 5, 20, 20))},
            {"mask_id": "s1", "mask": box_mask((40, 40, 42, 42))},
            {"mask_id": "s2", "mask": box_mask((43, 40, 45, 42))},
            {"mask_id": "s3", "mask": box_mask((40, 43, 42, 45))},
            {"mask_id": "isolated", "mask": box_mask((58, 58, 59, 59))},
        ]

        normalized, summary = normalize_candidate_masks(
            raw_masks,
            image_shape=(64, 64),
            config=CandidateMaskNormalizationConfig(
                max_mask_area_ratio=0.5,
                min_mask_area_ratio=0.0001,
                small_cluster_area_ratio=0.002,
                min_cluster_members=3,
                min_cluster_union_area_ratio=0.001,
            ),
        )

        mask_ids = [item["mask_id"] for item in normalized]
        self.assertIn("medium", mask_ids)
        self.assertTrue(any(str(mask_id).startswith("stuff_cluster::") for mask_id in mask_ids))
        stuff = next(item for item in normalized if str(item["mask_id"]).startswith("stuff_cluster::"))
        medium = next(item for item in normalized if item["mask_id"] == "medium")
        self.assertTrue(stuff["use_for_patch_graph"])
        self.assertFalse(stuff["use_as_component_node"])
        self.assertTrue(medium["use_for_patch_graph"])
        self.assertTrue(medium["use_as_component_node"])
        self.assertNotIn("huge", mask_ids)
        self.assertEqual(summary["num_excluded_large"], 1)
        self.assertEqual(summary["num_excluded_small"], 1)


if __name__ == "__main__":
    unittest.main()
