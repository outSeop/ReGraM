from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orchestration.notebook_orchestration import (  # noqa: E402
    build_baseline_specs,
    build_requested_run_configs,
    discover_manifest_names,
    expand_model_variants,
    model_config_from_experiment_config,
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

    def test_build_baseline_specs_maps_model_config_to_runner_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_baseline_specs(
                repo_root=root,
                exp_root=root / "exp",
                raw_loco_root=root / "raw",
                univad_caption_data_root=root / "caption",
                default_patchcore_device="cpu",
                model_config={
                    "patchcore": {
                        "sampler_percentage": 0.01,
                        "export_heatmaps": False,
                    },
                    "univad": {
                        "k_shot": 5,
                        "round": 2,
                        "heatmap_max_images": 4,
                    },
                },
            )

        patchcore_args = specs["PatchCore"]["extra_args"]
        univad_args = specs["UniVAD"]["extra_args"]

        self.assertIn("--sampler-percentage", patchcore_args)
        self.assertIn("0.01", patchcore_args)
        self.assertNotIn("--export-heatmaps", patchcore_args)
        self.assertIn("--k-shot", univad_args)
        self.assertIn("5", univad_args)
        self.assertIn("--round", univad_args)
        self.assertIn("2", univad_args)
        self.assertIn("--heatmap-max-images", univad_args)
        self.assertIn("4", univad_args)

    def test_build_baseline_specs_rejects_unknown_model_config_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Unknown model_config baseline"):
                build_baseline_specs(
                    repo_root=root,
                    exp_root=root / "exp",
                    raw_loco_root=root / "raw",
                    univad_caption_data_root=root / "caption",
                    default_patchcore_device="cpu",
                    model_config={"Unknown": {}},
                )

    def test_model_config_rejects_list_values_outside_sweep(self) -> None:
        with self.assertRaisesRegex(ValueError, "Put multiple values under `sweep.models"):
            model_config_from_experiment_config(
                {
                    "models": {
                        "univad": {
                            "k_shot": [1, 4],
                        }
                    }
                }
            )

    def test_expand_model_variants_builds_grid(self) -> None:
        resolved = model_config_from_experiment_config(
            {
                "models": {
                    "univad": {
                        "k_shot": 1,
                        "round": 0,
                    }
                }
            }
        )
        variants = expand_model_variants(
            resolved,
            {
                "enabled": True,
                "models": {
                    "univad": {
                        "k_shot": [1, 4],
                        "round": [0, 1, 2],
                    }
                },
            },
        )

        self.assertEqual(len(variants["UniVAD"]), 6)
        self.assertEqual(variants["UniVAD"][0]["id"], "k_shot_1__round_0")
        self.assertEqual(variants["UniVAD"][-1]["id"], "k_shot_4__round_2")
        self.assertEqual(variants["PatchCore"][0]["id"], "default")

    def test_expand_model_variants_rejects_unknown_sweep_baseline(self) -> None:
        resolved = model_config_from_experiment_config({"models": {}})
        with self.assertRaisesRegex(ValueError, "Unknown sweep model baseline"):
            expand_model_variants(
                resolved,
                {
                    "enabled": True,
                    "models": {
                        "unknown": {
                            "k": [1],
                        }
                    },
                },
            )

    def test_build_requested_run_configs_expands_sweep_to_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "query_position_shift.jsonl"
            manifest_path.write_text("", encoding="utf-8")
            specs = build_baseline_specs(
                repo_root=root,
                exp_root=root / "exp",
                raw_loco_root=root / "raw",
                univad_caption_data_root=root / "caption",
                default_patchcore_device="cpu",
                model_config={"univad": {"k_shot": 1, "round": 0}},
                model_sweep={
                    "enabled": True,
                    "models": {"univad": {"k_shot": [1, 4], "round": [0, 1]}},
                },
            )

            configs = build_requested_run_configs(
                active_baselines=["UniVAD"],
                specs=specs,
                categories=["breakfast_box"],
                manifest_paths=[manifest_path],
                manifest_names=[manifest_path.name],
                report_root=root / "reports",
                wandb_project="project",
                wandb_mode="disabled",
                selected_severities=["high"],
            )

        self.assertEqual(len(configs), 4)
        self.assertEqual(configs[0]["model_variant_id"], "k_shot_1__round_0")
        self.assertIn("k_shot_1__round_0", str(configs[0]["summary_path"]))
        self.assertIn("--output", configs[0]["runner_cmd"])
        self.assertIn("--log-dir", configs[0]["runner_cmd"])
        self.assertIn("UniVAD:k_shot_4__round_1", [config["display_baseline"] for config in configs])


if __name__ == "__main__":
    unittest.main()
