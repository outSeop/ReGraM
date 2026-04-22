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
    build_common_run_config,
    prepare_manifest_shift_run_spec,
    prepare_output_paths,
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


if __name__ == "__main__":
    unittest.main()
