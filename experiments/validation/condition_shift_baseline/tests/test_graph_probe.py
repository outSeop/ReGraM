from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_probe.component_io import component_quality_row, load_component_nodes  # noqa: E402
from graph_probe.component_matching import match_components  # noqa: E402
from graph_probe.batch_compare import (  # noqa: E402
    compare_summary_conditions,
    row_identity,
    summarize_condition_scores,
)
from graph_probe.graph_consistency import graph_consistency_score  # noqa: E402
from graph_probe.graph_features import edge_feature  # noqa: E402
from graph_probe.run_logical_probe import run_logical_probe  # noqa: E402
from graph_probe.run_stability_probe import run_stability_probe  # noqa: E402


def node(
    node_id: str,
    bbox: tuple[int, int, int, int],
    *,
    node_type: str = "thing",
    mask_ids: list[str] | None = None,
) -> dict:
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return {
        "node_id": node_id,
        "node_type": node_type,
        "source": "test",
        "mask_ids": mask_ids or [node_id],
        "area": area,
        "area_ratio": area / 10_000,
        "bbox": [x1, y1, x2, y2],
        "centroid": [(x1 + x2) / 2, (y1 + y2) / 2],
        "mean_rgb": [100.0, 100.0, 100.0],
        "std_rgb": [1.0, 1.0, 1.0],
        "mean_lab": None,
        "std_lab": None,
        "num_members": len(mask_ids or [node_id]),
        "member_stats": {},
        "confidence": None,
        "debug": {},
    }


class GraphProbeTests(unittest.TestCase):
    def test_matching_is_geometry_based_and_tracks_unmatched(self) -> None:
        ref_nodes = [
            node("a", (0, 0, 10, 10)),
            node("b", (60, 0, 70, 10)),
        ]
        query_nodes = [
            node("extra", (20, 80, 30, 90)),
            node("b2", (61, 1, 71, 11)),
            node("a2", (1, 1, 11, 11)),
        ]

        result = match_components(ref_nodes, query_nodes, image_size=(100, 100))

        self.assertEqual([(left, right) for left, right, _ in result.matches], [(0, 2), (1, 1)])
        self.assertEqual(result.unmatched_ref, [])
        self.assertEqual(result.unmatched_query, [0])

    def test_edge_feature_and_consistency_are_stable_for_small_shift(self) -> None:
        ref_nodes = [
            node("a", (0, 0, 10, 10)),
            node("b", (60, 0, 70, 10)),
            node("c", (20, 60, 35, 75), node_type="stuff_cluster", mask_ids=["c1", "c2"]),
        ]
        query_nodes = [
            node("a2", (1, 0, 11, 10)),
            node("b2", (61, 0, 71, 10)),
            node("c2", (21, 60, 36, 75), node_type="stuff_cluster", mask_ids=["c1", "c2"]),
        ]

        edge = edge_feature(ref_nodes[0], ref_nodes[1], image_size=(100, 100))
        score = graph_consistency_score(ref_nodes, query_nodes, image_size=(100, 100))

        self.assertAlmostEqual(edge["dx"], 0.6)
        self.assertEqual(score["S_unmatched"], 0.0)
        self.assertEqual(score["matched_node_count"], 3)
        self.assertLess(score["S_edge"], 0.05)

    def test_logical_change_increases_edge_or_unmatched_score(self) -> None:
        ref_nodes = [
            node("a", (0, 0, 10, 10)),
            node("b", (60, 0, 70, 10)),
            node("c", (20, 60, 35, 75)),
        ]
        logical_nodes = [
            node("a2", (0, 0, 10, 10)),
            node("b_misplaced", (15, 0, 25, 10)),
            node("extra", (80, 80, 90, 90)),
        ]

        score = graph_consistency_score(ref_nodes, logical_nodes, image_size=(100, 100))

        self.assertGreater(score["S_edge"], 0.1)

    def test_component_quality_row_counts_node_types_and_raw_masks(self) -> None:
        nodes = [
            node("a", (0, 0, 10, 10), node_type="thing"),
            node("stuff", (20, 20, 40, 40), node_type="stuff_cluster", mask_ids=["s1", "s2"]),
            node("small", (80, 80, 85, 85), node_type="small_isolated"),
        ]

        row = component_quality_row(
            nodes=nodes,
            image_id="000.png",
            category="breakfast_box",
            split="query",
            shift_type="brightness",
            severity="high",
        )

        self.assertEqual(row["num_thing"], 1)
        self.assertEqual(row["num_stuff_cluster"], 1)
        self.assertEqual(row["num_small_isolated"], 1)
        self.assertEqual(row["num_raw_masks"], 4)

    def test_stability_and_logical_runners_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            reference = tmp_path / "reference.json"
            shifted = tmp_path / "shifted.json"
            logical = tmp_path / "logical.json"
            ref_nodes = [
                node("a", (0, 0, 10, 10)),
                node("b", (60, 0, 70, 10)),
            ]
            shifted_nodes = [
                node("a2", (1, 0, 11, 10)),
                node("b2", (61, 0, 71, 10)),
            ]
            logical_nodes = [
                node("a2", (0, 0, 10, 10)),
                node("b_misplaced", (20, 40, 30, 50)),
                node("extra", (80, 80, 90, 90)),
            ]
            reference.write_text(json.dumps({"rows": [_row("000.png", ref_nodes)]}), encoding="utf-8")
            shifted.write_text(json.dumps({"rows": [_row("000.png", shifted_nodes)]}), encoding="utf-8")
            logical.write_text(json.dumps({"rows": [_row("000.png", logical_nodes)]}), encoding="utf-8")

            stability = run_stability_probe(
                reference_path=reference,
                query_paths=[shifted],
                category="breakfast_box",
                conditions=["brightness_high"],
                output_json=tmp_path / "stability.json",
                output_csv=tmp_path / "stability.csv",
                quality_csv=tmp_path / "quality.csv",
            )
            logical_result = run_logical_probe(
                reference_path=reference,
                normal_query_path=shifted,
                logical_query_path=logical,
                category="breakfast_box",
                output_json=tmp_path / "logical.json.out",
                output_csv=tmp_path / "logical.csv",
            )

            self.assertEqual(len(stability["rows"]), 1)
            self.assertEqual(len(logical_result["rows"]), 2)
            self.assertTrue((tmp_path / "stability.csv").exists())
            self.assertTrue((tmp_path / "logical.csv").exists())
            self.assertEqual(load_component_nodes(reference, row_index=0), ref_nodes)

    def test_batch_compare_matches_base_image_id_and_writes_condition_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            reference = tmp_path / "clean.json"
            shifted = tmp_path / "shifted.json"
            logical = tmp_path / "logical.json"
            ref_nodes = [
                node("a", (0, 0, 10, 10)),
                node("b", (60, 0, 70, 10)),
            ]
            shifted_nodes = [
                node("a_shift", (1, 0, 11, 10)),
                node("b_shift", (61, 0, 71, 10)),
            ]
            logical_nodes = [
                node("a_logical", (0, 0, 10, 10)),
                node("b_moved", (20, 40, 30, 50)),
            ]
            reference.write_text(
                json.dumps({"rows": [_row_with_base("000.png", "clean_000.png", ref_nodes)]}),
                encoding="utf-8",
            )
            shifted.write_text(
                json.dumps({"rows": [_row_with_base("000.png", "brightness_000.png", shifted_nodes)]}),
                encoding="utf-8",
            )
            logical.write_text(
                json.dumps({"rows": [_row_with_base("000.png", "logical_000.png", logical_nodes)]}),
                encoding="utf-8",
            )

            score_df = compare_summary_conditions(
                reference_summary_path=reference,
                query_summary_paths={
                    "brightness_high": shifted,
                    "logical_anomaly": logical,
                },
                category="breakfast_box",
                output_csv=tmp_path / "scores.csv",
            )
            summary_df = summarize_condition_scores(score_df, output_csv=tmp_path / "summary.csv")

            self.assertEqual(len(score_df), 2)
            self.assertEqual(set(score_df["condition"]), {"brightness_high", "logical_anomaly"})
            self.assertEqual(set(score_df["image_id"]), {"000.png"})
            self.assertEqual(row_identity({"source_path": "/tmp/fallback.png"}), "fallback.png")
            self.assertTrue((tmp_path / "scores.csv").exists())
            self.assertTrue((tmp_path / "summary.csv").exists())
            self.assertEqual(set(summary_df["condition"]), {"brightness_high", "logical_anomaly"})
            self.assertIn("S_edge_mean", summary_df.columns)


def _row(source_id: str, nodes: list[dict]) -> dict:
    return {
        "source_id": source_id,
        "image_width": 100,
        "image_height": 100,
        "raw_mask_count": len({mask_id for node_item in nodes for mask_id in node_item["mask_ids"]}),
        "component_nodes": nodes,
    }


def _row_with_base(base_image_id: str, source_id: str, nodes: list[dict]) -> dict:
    row = _row(source_id, nodes)
    row["base_image_id"] = base_image_id
    return row


if __name__ == "__main__":
    unittest.main()
