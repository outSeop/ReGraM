from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


CORE_DIR = Path(__file__).resolve().parents[1] / "src" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from manifest_shift_common import (  # noqa: E402
    THRESHOLD_POLICY_CLEAN_MAX,
    build_clean_metric_snapshot,
    build_common_run_config,
    build_results_scaffold,
    build_shift_metric_snapshot,
    prepare_manifest_shift_run_spec,
    prepare_output_paths,
    summarize_scores,
)
from repo_paths import REPO_ROOT  # noqa: E402


class ManifestShiftCommonTests(unittest.TestCase):
    def test_single_manifest_spec(self) -> None:
        run_spec = prepare_manifest_shift_run_spec(
            category="breakfast_box",
            input_root="data/query_normal_clean",
            manifest_paths=["manifests/query_motion_blur.jsonl"],
        )

        self.assertEqual(run_spec.manifest_name, "query_motion_blur.jsonl")
        self.assertEqual(run_spec.shift_family, "motion_blur")
        self.assertEqual(run_spec.severity_label, "all")
        self.assertEqual(run_spec.output_suffix, "motion_blur_all")
        self.assertEqual(run_spec.selected_severities, [])
        self.assertEqual(run_spec.severity_spec, {"mixed": True})
        self.assertEqual(run_spec.augmentation_types_seen, ["motion_blur"])
        self.assertGreater(run_spec.total_entries, 0)

    def test_multi_manifest_spec(self) -> None:
        run_spec = prepare_manifest_shift_run_spec(
            category="breakfast_box",
            input_root="data/query_normal_clean",
            manifest_paths=[
                "manifests/query_brightness.jsonl",
                "manifests/query_low_light.jsonl",
            ],
        )

        self.assertEqual(
            run_spec.manifest_repr,
            "multi:query_brightness.jsonl,query_low_light.jsonl",
        )
        self.assertEqual(run_spec.manifest_name, run_spec.manifest_repr)
        self.assertEqual(run_spec.shift_family, "multi")
        self.assertEqual(run_spec.augmentation_types_seen, ["brightness", "low_light"])
        self.assertEqual(run_spec.severity_label, "all")
        self.assertGreater(run_spec.total_entries, 0)

    def test_in_memory_manifest_spec(self) -> None:
        temp_dir = Path(
            tempfile.mkdtemp(prefix="manifest_shift_common_", dir=REPO_ROOT)
        )
        try:
            category_dir = temp_dir / "toy_widget"
            category_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(category_dir / "000.png")

            run_spec = prepare_manifest_shift_run_spec(
                category="toy_widget",
                input_root=temp_dir,
                augmentation_type="compression",
            )

            self.assertEqual(run_spec.manifest_repr, "in_memory:compression")
            self.assertEqual(run_spec.shift_family, "compression")
            self.assertEqual(run_spec.augmentation_types_seen, ["compression"])
            self.assertEqual(run_spec.total_entries, 3)
        finally:
            shutil.rmtree(temp_dir)

    def test_severity_filter_and_spec(self) -> None:
        run_spec = prepare_manifest_shift_run_spec(
            category="breakfast_box",
            input_root="data/query_normal_clean",
            manifest_paths=["manifests/query_motion_blur.jsonl"],
            severities=["low"],
        )

        self.assertEqual(run_spec.selected_severities, ["low"])
        self.assertEqual(run_spec.severity_label, "low")
        self.assertEqual(run_spec.output_suffix, "motion_blur_low")
        self.assertEqual(run_spec.severity_spec, {"kernel_size": 5})
        self.assertEqual(run_spec.severity_spec_flat, {"severity_param_kernel_size": 5})
        self.assertEqual(list(run_spec.grouped_entries["motion_blur"]), ["low"])

    def test_severity_filter_without_match_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No manifest entries matched"):
            prepare_manifest_shift_run_spec(
                category="breakfast_box",
                input_root="data/query_normal_clean",
                manifest_paths=["manifests/query_motion_blur.jsonl"],
                severities=["none"],
            )

    def test_common_config_and_output_paths(self) -> None:
        run_spec = prepare_manifest_shift_run_spec(
            category="breakfast_box",
            input_root="data/query_normal_clean",
            manifest_paths=["manifests/query_motion_blur.jsonl"],
            severities=["low"],
        )

        config = build_common_run_config(
            run_spec=run_spec,
            input_root="data/query_normal_clean",
            data_root="data/row/mvtec_loco_anomaly_detection",
            device="cuda",
            wandb_log_images=True,
            wandb_max_images=2,
            extra_config={"image_size": 448},
        )
        output_paths = prepare_output_paths(
            output_root="experiments/validation/condition_shift_baseline/reports/univad_manifest_shift",
            log_dir="experiments/validation/condition_shift_baseline/reports/univad_manifest_shift/logs",
            category="breakfast_box",
            output_suffix=run_spec.output_suffix,
        )

        self.assertEqual(config["threshold_policy"], THRESHOLD_POLICY_CLEAN_MAX)
        self.assertEqual(config["severity"], "low")
        self.assertEqual(config["severity_label"], "low")
        self.assertEqual(config["severity_param_kernel_size"], 5)
        self.assertTrue(str(output_paths.output_path).endswith("breakfast_box_motion_blur_low.json"))
        self.assertTrue(str(output_paths.log_path).endswith("breakfast_box_motion_blur_low.log.txt"))

    def test_summarize_scores_includes_distribution_stats(self) -> None:
        summary = summarize_scores([0.1, 0.2, 0.4, 0.8], threshold=0.3)

        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["median"], 0.3, places=6)
        self.assertAlmostEqual(summary["p25"], 0.175, places=6)
        self.assertAlmostEqual(summary["p75"], 0.5, places=6)
        self.assertAlmostEqual(summary["p95"], 0.74, places=6)
        self.assertAlmostEqual(summary["fpr_over_clean_max"], 0.5, places=6)

    def test_shift_metric_snapshot_aggregates_new_metrics(self) -> None:
        results = {
            "augmentations": {
                "brightness": {
                    "low": {
                        "shifted_normal_fpr": 0.2,
                        "image_auroc_drop_from_clean": 1.5,
                        "mean_score_shift": 0.1,
                        "median_score_shift": 0.08,
                    },
                },
                "motion_blur": {
                    "high": {
                        "shifted_normal_fpr": 0.6,
                        "image_auroc_drop_from_clean": 4.0,
                        "mean_score_shift": 0.3,
                        "median_score_shift": 0.25,
                    },
                },
            }
        }

        snapshot = build_shift_metric_snapshot(results)

        self.assertAlmostEqual(snapshot["mean_shifted_normal_fpr"], 0.4, places=6)
        self.assertAlmostEqual(snapshot["worst_shifted_normal_fpr"], 0.6, places=6)
        self.assertEqual(snapshot["worst_shift_cell_by_fpr"], "motion_blur/high")
        self.assertAlmostEqual(snapshot["mean_image_auroc_drop_from_clean"], 2.75, places=6)
        self.assertAlmostEqual(snapshot["worst_image_auroc_drop_from_clean"], 4.0, places=6)
        self.assertEqual(snapshot["worst_shift_cell_by_auroc_drop"], "motion_blur/high")
        self.assertAlmostEqual(snapshot["mean_score_shift"], 0.2, places=6)
        self.assertAlmostEqual(snapshot["mean_median_score_shift"], 0.165, places=6)
        self.assertAlmostEqual(snapshot["mean_fpr_by_severity/low"], 0.2, places=6)
        self.assertAlmostEqual(
            snapshot["mean_image_auroc_drop_from_clean_by_severity/high"],
            4.0,
            places=6,
        )

    def test_clean_metric_snapshot_includes_logical_structural_split(self) -> None:
        run_spec = prepare_manifest_shift_run_spec(
            category="breakfast_box",
            input_root="data/query_normal_clean",
            manifest_paths=["manifests/query_motion_blur.jsonl"],
        )
        results = build_results_scaffold(
            updated_at="2026-05-03 00:00:00 KST",
            category="breakfast_box",
            run_spec=run_spec,
            clean_good=summarize_scores([0.1, 0.2], threshold=0.2),
            clean_anomaly=summarize_scores([0.8, 0.9, 1.0], threshold=0.2),
            clean_image_auroc=99.0,
            clean_logical_anomaly=summarize_scores([0.8, 0.9], threshold=0.2),
            clean_structural_anomaly=summarize_scores([1.0], threshold=0.2),
            clean_logical_image_auroc=98.0,
            clean_structural_image_auroc=100.0,
        )

        snapshot = build_clean_metric_snapshot(results)

        self.assertEqual(snapshot["clean_logical_image_auroc"], 98.0)
        self.assertEqual(snapshot["clean_structural_image_auroc"], 100.0)
        self.assertAlmostEqual(snapshot["clean_logical_anomaly_mean"], 0.85)
        self.assertAlmostEqual(snapshot["clean_structural_anomaly_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
