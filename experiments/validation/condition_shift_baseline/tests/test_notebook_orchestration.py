from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orchestration.notebook_orchestration import (  # noqa: E402
    build_requested_run_configs,
    discover_manifest_names,
    normalize_selected_severities,
)


class NotebookOrchestrationTests(unittest.TestCase):
    def test_discover_manifest_names_prefers_configured_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_root = Path(tmp) / "manifests"
            manifest_root.mkdir()
            for name in (
                "query_motion_blur.jsonl",
                "query_low_light.jsonl",
                "query_gaussian_noise.jsonl",
                "query_brightness.jsonl",
                "query_identity.jsonl",
            ):
                (manifest_root / name).write_text("", encoding="utf-8")

            manifest_names = discover_manifest_names(
                manifest_roots=[manifest_root],
                configured_names=[
                    "query_motion_blur.jsonl",
                    "query_low_light.jsonl",
                    "query_gaussian_noise.jsonl",
                ],
                auto_discover=True,
                excluded_names={"query_identity.jsonl"},
            )

        self.assertEqual(
            manifest_names,
            [
                "query_motion_blur.jsonl",
                "query_low_light.jsonl",
                "query_gaussian_noise.jsonl",
            ],
        )

    def test_discover_manifest_names_auto_discovers_when_config_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_root = Path(tmp) / "manifests"
            manifest_root.mkdir()
            for name in (
                "query_motion_blur.jsonl",
                "query_low_light.jsonl",
                "query_identity.jsonl",
                "notes.jsonl",
            ):
                (manifest_root / name).write_text("", encoding="utf-8")

            manifest_names = discover_manifest_names(
                manifest_roots=[manifest_root],
                configured_names=[],
                auto_discover=True,
                excluded_names={"query_identity.jsonl"},
            )

        self.assertEqual(manifest_names, ["query_low_light.jsonl", "query_motion_blur.jsonl"])

    def test_normalize_selected_severities_accepts_middle_alias(self) -> None:
        self.assertEqual(normalize_selected_severities(["low", "middle", "high"]), ["low", "medium", "high"])
        self.assertEqual(normalize_selected_severities([]), [])
        self.assertEqual(normalize_selected_severities(["all"]), [])

    def test_build_requested_run_configs_adds_severity_filter_and_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "query_gaussian_noise.jsonl"
            manifest_path.write_text("", encoding="utf-8")
            specs = {
                "PatchCore": {
                    "runner_name": "PatchCore",
                    "runner_path": root / "runner.py",
                    "runner_inputs": "-",
                    "runner_outputs": "-",
                    "report_subdir": "patchcore_manifest_shift",
                    "data_root": root / "data",
                    "device": "cpu",
                    "extra_args": [],
                    "use_wandb": False,
                    "wandb_group": "patchcore",
                    "wandb_log_images": False,
                    "wandb_max_images": 0,
                    "notes": "-",
                }
            }

            configs = build_requested_run_configs(
                active_baselines=["PatchCore"],
                specs=specs,
                categories=["breakfast_box"],
                manifest_paths=[manifest_path],
                manifest_names=[manifest_path.name],
                report_root=root / "reports",
                wandb_project="project",
                wandb_mode="disabled",
                selected_severities=["middle"],
            )

        self.assertEqual(configs[0]["selected_severities"], ["medium"])
        self.assertEqual(configs[0]["summary_path"].name, "breakfast_box_gaussian_noise_medium.json")
        self.assertIn("--severities", configs[0]["runner_cmd"])
        self.assertIn("medium", configs[0]["runner_cmd"])


if __name__ == "__main__":
    unittest.main()
