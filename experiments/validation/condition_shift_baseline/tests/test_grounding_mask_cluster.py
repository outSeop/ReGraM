from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from relation.grounding_mask_cluster import (  # noqa: E402
    MaskFeature,
    bbox_gap,
    build_small_mask_adjacency,
    cluster_masks_to_components,
    cluster_grounding_masks,
    color_distance,
    compute_mask_features,
    connected_components,
    raw_masks_from_label_image,
    resolve_cluster_config,
    summarize_components,
)
from relation.run_relation_probe import run_probe  # noqa: E402


def box_mask(box: tuple[int, int, int, int], *, shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class GroundingMaskClusterTests(unittest.TestCase):
    def test_public_api_computes_features_clusters_and_summarizes(self) -> None:
        image = np.full((100, 100, 3), 240, dtype=np.uint8)
        raw_masks = [
            {"mask_id": "large", "mask": box_mask((5, 5, 25, 25)), "score": 0.9},
            {"mask_id": "chip_a", "mask": box_mask((50, 50, 55, 55)), "score": 0.8},
            {"mask_id": "chip_b", "mask": box_mask((58, 50, 63, 55)), "score": 0.7},
            {"mask_id": "chip_c", "mask": box_mask((66, 50, 71, 55)), "score": 0.6},
        ]

        features = compute_mask_features(image, raw_masks, {"use_lab_color": False})
        self.assertTrue(all(isinstance(feature, MaskFeature) for feature in features))
        small_features = [feature for feature in features if feature.mask_id != "large"]
        adjacency = build_small_mask_adjacency(small_features, image.shape[:2])
        self.assertEqual(connected_components(adjacency), [[0, 1, 2]])
        self.assertEqual(bbox_gap(small_features[0].bbox, small_features[1].bbox), 3.0)
        self.assertEqual(color_distance(small_features[0], small_features[1], use_lab=False), 0.0)

        nodes = cluster_masks_to_components(image, raw_masks, config={"use_lab_color": False})
        summary = summarize_components(nodes)

        self.assertEqual(summary["num_nodes"], 2)
        self.assertEqual(summary["num_thing"], 1)
        self.assertEqual(summary["num_stuff_cluster"], 1)
        self.assertEqual(summary["num_raw_masks_used"], 4)

    def test_clusters_nearby_small_masks_into_stuff_node(self) -> None:
        image = np.full((100, 100, 3), 240, dtype=np.uint8)
        raw_masks = [
            {"mask_id": "large", "mask": box_mask((5, 5, 25, 25)), "score": 0.9},
            {"mask_id": "chip_a", "mask": box_mask((50, 50, 55, 55)), "score": 0.8},
            {"mask_id": "chip_b", "mask": box_mask((58, 50, 63, 55)), "score": 0.7},
            {"mask_id": "chip_c", "mask": box_mask((66, 50, 71, 55)), "score": 0.6},
        ]

        nodes = cluster_grounding_masks(image, raw_masks)

        node_types = [node["node_type"] for node in nodes]
        self.assertIn("thing", node_types)
        self.assertIn("stuff_cluster", node_types)
        stuff = next(node for node in nodes if node["node_type"] == "stuff_cluster")
        self.assertEqual(stuff["mask_ids"], ["chip_a", "chip_b", "chip_c"])
        self.assertEqual(stuff["num_members"], 3)
        self.assertEqual(stuff["member_stats"]["member_areas"], [25, 25, 25])
        self.assertGreater(stuff["member_stats"]["density"], 0)
        self.assertEqual(stuff["debug"]["reason"], "small_masks_clustered")
        json.dumps(nodes)

    def test_filters_oversized_masks_when_max_area_ratio_is_set(self) -> None:
        image = np.full((100, 100, 3), 240, dtype=np.uint8)
        raw_masks = [
            {"mask_id": "background", "mask": box_mask((0, 0, 90, 90))},
            {"mask_id": "object", "mask": box_mask((10, 10, 30, 30))},
        ]

        nodes = cluster_grounding_masks(
            image,
            raw_masks,
            config={
                "use_lab_color": False,
                "max_mask_area_ratio": 0.55,
            },
        )

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["mask_ids"], ["object"])
        self.assertEqual(nodes[0]["node_type"], "thing")

    def test_preserves_small_masks_when_cluster_conditions_fail(self) -> None:
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        raw_masks = [
            {"mask_id": "part_a", "mask": box_mask((10, 10, 15, 15))},
            {"mask_id": "part_b", "mask": box_mask((18, 10, 23, 15))},
        ]

        nodes = cluster_grounding_masks(image, raw_masks)

        self.assertEqual([node["node_type"] for node in nodes], ["small_isolated", "small_isolated"])
        self.assertEqual(
            {node["debug"]["reason"] for node in nodes},
            {"not_enough_cluster_members"},
        )

    def test_can_absorb_nearby_singleton_into_existing_stuff_cluster(self) -> None:
        image = np.full((100, 100, 3), 240, dtype=np.uint8)
        raw_masks = [
            {"mask_id": "chip_a", "mask": box_mask((50, 50, 55, 55))},
            {"mask_id": "chip_b", "mask": box_mask((58, 50, 63, 55))},
            {"mask_id": "chip_c", "mask": box_mask((66, 50, 71, 55))},
            {"mask_id": "chip_outlier", "mask": box_mask((82, 53, 87, 58))},
        ]

        nodes = cluster_grounding_masks(
            image,
            raw_masks,
            config={
                "absorb_nearby_singletons": True,
                "absorb_max_centroid_dist_ratio": 0.2,
                "absorb_max_bbox_gap_ratio": 0.2,
                "max_centroid_dist_ratio": 0.08,
                "max_bbox_gap_ratio": 0.03,
            },
        )

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["node_type"], "stuff_cluster")
        self.assertEqual(nodes[0]["num_members"], 4)
        self.assertIn("chip_outlier", nodes[0]["mask_ids"])
        self.assertEqual(nodes[0]["debug"]["cluster_conditions"]["absorbed_nearby_group_count"], 1)

    def test_same_label_can_cluster_even_when_color_differs(self) -> None:
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        masks = [
            box_mask((50, 50, 55, 55)),
            box_mask((58, 50, 63, 55)),
            box_mask((66, 50, 71, 55)),
        ]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for mask, color in zip(masks, colors, strict=True):
            image[mask] = color
        raw_masks = [
            {"mask_id": f"part_{index}", "mask": mask, "label": "banana_chip"}
            for index, mask in enumerate(masks)
        ]

        nodes = cluster_grounding_masks(
            image,
            raw_masks,
            config={"use_lab_color": False, "max_color_dist": 1.0},
        )

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["node_type"], "stuff_cluster")
        self.assertEqual(nodes[0]["debug"]["cluster_conditions"]["edge_feature_reasons"], ["same_label"])

    def test_label_image_is_split_into_raw_masks_without_treating_color_as_class(self) -> None:
        label_image = np.zeros((10, 10, 3), dtype=np.uint8)
        label_image[1:3, 1:3] = [255, 0, 0]
        label_image[5:8, 5:8] = [0, 255, 0]

        raw_masks = raw_masks_from_label_image(label_image)

        self.assertEqual(len(raw_masks), 2)
        self.assertEqual({mask["source"] for mask in raw_masks}, {"label_image"})
        self.assertNotIn("label", raw_masks[0])

    def test_category_config_overrides_default_thresholds(self) -> None:
        config = resolve_cluster_config(
            {
                "default": {"large_area_ratio": 0.02},
                "categories": {
                    "breakfast_box": {
                        "large_area_ratio": 0.04,
                        "absorb_nearby_singletons": True,
                    }
                },
            },
            category="breakfast_box",
        )

        self.assertEqual(config["large_area_ratio"], 0.04)
        self.assertTrue(config["absorb_nearby_singletons"])

    def test_relation_probe_can_use_colored_grounding_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            category = "breakfast_box"
            image_path = repo_root / "data" / "row" / "mvtec_loco_anomaly_detection" / category / "test" / "good" / "000.png"
            image_path.parent.mkdir(parents=True)
            image = np.full((100, 100, 3), 240, dtype=np.uint8)
            Image.fromarray(image).save(image_path)

            mask_path = (
                repo_root
                / "external"
                / "UniVAD"
                / "masks"
                / "mvtec_loco_caption"
                / category
                / "test"
                / "good"
                / "000"
                / "grounding_mask.png"
            )
            mask_path.parent.mkdir(parents=True)
            label_mask = np.zeros((100, 100, 3), dtype=np.uint8)
            label_mask[5:25, 5:25] = [255, 0, 0]
            label_mask[50:55, 50:55] = [0, 255, 0]
            label_mask[50:55, 58:63] = [0, 0, 255]
            label_mask[50:55, 66:71] = [255, 255, 0]
            Image.fromarray(label_mask).save(mask_path)

            manifest_path = repo_root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_path": str(image_path),
                        "source_path_mode": "absolute",
                        "category": category,
                        "source_id": "000.png",
                        "augmentation_type": "position_shift",
                        "severity": "high",
                        "seed": 1,
                        "params": {"scale": 0.7, "placement": "bottom_right"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = repo_root / "out.json"

            summary = run_probe(
                repo_root=repo_root,
                manifest_path=manifest_path,
                category=category,
                severity="high",
                limit=1,
                output_path=output_path,
                max_components=8,
                component_model="grounding_mask_cluster",
                grounding_mask_root=repo_root / "external" / "UniVAD" / "masks" / "mvtec_loco_caption",
                grounding_mask_data_root=repo_root / "data" / "mvtec_loco_caption",
                grounding_cluster_config=None,
                progress_every=0,
            )

            self.assertEqual(summary["evaluated_count"], 1)
            row = summary["rows"][0]
            self.assertEqual(row["component_source"], "grounding_mask_cluster")
            self.assertEqual(row["stuff_cluster_count"], 1)
            self.assertEqual(row["thing_count"], 1)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
