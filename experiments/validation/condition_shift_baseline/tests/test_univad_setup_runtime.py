from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
for import_dir in (SRC_DIR, SRC_DIR / "core", SRC_DIR / "orchestration"):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from univad import setup_runtime as setup_runtime_module  # noqa: E402
from univad.setup_runtime import (  # noqa: E402
    _dependency_restart_note,
    checkpoint_download_urls,
    checkpoint_file_ready,
    collect_univad_missing_mask_image_paths,
    collect_missing_checkpoint_files,
    default_univad_setup_flags,
    install_univad_requirements_without_torch,
    maybe_backup_univad_checkpoints_to_drive,
    maybe_prepare_univad_grounding_masks,
    maybe_restore_univad_checkpoints_from_drive,
    write_univad_runtime_constraints,
)
from univad.transformers_runtime import disable_transformers_tensorflow_backend  # noqa: E402


class UniVADSetupRuntimeTests(unittest.TestCase):
    def test_default_model_checkpoint_drive_root_is_research_shared(self) -> None:
        flags = default_univad_setup_flags()

        self.assertEqual(
            flags["model_checkpoint_drive_backup_root"],
            "/content/drive/MyDrive/ReGraM/model_checkpoints",
        )

    def test_checkpoint_file_ready_requires_expected_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            checkpoint_path = checkpoint_root / "toy.pth"
            checkpoint_path.write_bytes(b"partial")
            spec = {
                "checkpoint_root": checkpoint_root,
                "required_checkpoint_files": ["toy.pth"],
                "checkpoint_expected_bytes": {"toy.pth": 10},
            }

            self.assertFalse(checkpoint_file_ready(spec, "toy.pth"))
            self.assertEqual(collect_missing_checkpoint_files(spec), ["toy.pth"])

            checkpoint_path.write_bytes(b"0123456789")

            self.assertTrue(checkpoint_file_ready(spec, "toy.pth"))
            self.assertEqual(collect_missing_checkpoint_files(spec), [])

    def test_checkpoint_download_urls_accepts_string_or_list(self) -> None:
        self.assertEqual(
            checkpoint_download_urls(
                {"checkpoint_download_urls": {"a.pth": "https://example.test/a.pth"}},
                "a.pth",
            ),
            ["https://example.test/a.pth"],
        )
        self.assertEqual(
            checkpoint_download_urls(
                {
                    "checkpoint_download_urls": {
                        "b.pth": [
                            "https://example.test/b1.pth",
                            "https://example.test/b2.pth",
                        ]
                    }
                },
                "b.pth",
            ),
            ["https://example.test/b1.pth", "https://example.test/b2.pth"],
        )

    def test_checkpoint_drive_restore_copies_ready_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_root = root / "local_ckpts"
            drive_root = root / "drive_ckpts"
            drive_root.mkdir()
            (drive_root / "toy.pth").write_bytes(b"0123456789")
            spec = {
                "checkpoint_root": checkpoint_root,
                "required_checkpoint_files": ["toy.pth"],
                "checkpoint_expected_bytes": {"toy.pth": 10},
            }

            status = maybe_restore_univad_checkpoints_from_drive(
                spec,
                {
                    "restore_model_checkpoints_from_drive": True,
                    "auto_mount_google_drive_for_model_checkpoints": False,
                    "model_checkpoint_drive_backup_root": str(drive_root),
                },
            )

            self.assertEqual(status["status"], "ok")
            self.assertEqual(status["copied_checkpoint_files"], ["toy.pth"])
            self.assertTrue(checkpoint_file_ready(spec, "toy.pth"))

    def test_checkpoint_drive_backup_skips_already_cached_ready_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_root = root / "local_ckpts"
            drive_root = root / "drive_ckpts"
            checkpoint_root.mkdir()
            drive_root.mkdir()
            (checkpoint_root / "toy.pth").write_bytes(b"0123456789")
            (drive_root / "toy.pth").write_bytes(b"0123456789")
            spec = {
                "checkpoint_root": checkpoint_root,
                "required_checkpoint_files": ["toy.pth"],
                "checkpoint_expected_bytes": {"toy.pth": 10},
            }

            status = maybe_backup_univad_checkpoints_to_drive(
                spec,
                {
                    "backup_model_checkpoints_to_drive": True,
                    "auto_mount_google_drive_for_model_checkpoints": False,
                    "model_checkpoint_drive_backup_root": str(drive_root),
                },
            )

            self.assertEqual(status["status"], "already_present")
            self.assertEqual(status["copied_checkpoint_files"], [])
            self.assertEqual(status["cached_checkpoint_files"], ["toy.pth"])

    def test_disable_transformers_tensorflow_backend_patches_loaded_modules(self) -> None:
        fake_import_utils = types.SimpleNamespace(_tf_available=True, _tf_version="5.0")
        fake_generic = types.SimpleNamespace(is_tf_available=lambda: True)
        previous_import_utils = sys.modules.get("transformers.utils.import_utils")
        previous_generic = sys.modules.get("transformers.utils.generic")
        try:
            sys.modules["transformers.utils.import_utils"] = fake_import_utils
            sys.modules["transformers.utils.generic"] = fake_generic

            disable_transformers_tensorflow_backend()

            self.assertEqual(fake_import_utils._tf_available, False)
            self.assertEqual(fake_import_utils._tf_version, "N/A")
            self.assertEqual(fake_generic.is_tf_available(), False)
        finally:
            if previous_import_utils is None:
                sys.modules.pop("transformers.utils.import_utils", None)
            else:
                sys.modules["transformers.utils.import_utils"] = previous_import_utils
            if previous_generic is None:
                sys.modules.pop("transformers.utils.generic", None)
            else:
                sys.modules["transformers.utils.generic"] = previous_generic

    def test_dependency_restart_note_combines_all_restart_reasons(self) -> None:
        note = _dependency_restart_note(
            [
                ("runtime_dependency_unavailable", {"restart_required": True, "note": "runtime deps installed"}),
                ("torch_stack_unavailable", {"restart_required": True, "note": "torch stack installed"}),
                ("transformers_stack_unavailable", {"restart_required": False, "note": "already compatible"}),
            ]
        )

        self.assertIn("restart runtime once", note)
        self.assertIn("runtime_dependency_unavailable: runtime deps installed", note)
        self.assertIn("torch_stack_unavailable: torch stack installed", note)
        self.assertNotIn("already compatible", note)

    def test_univad_requirement_install_uses_runtime_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            requirements_path = Path(tmp) / "requirements.txt"
            requirements_path.write_text(
                "\n".join(
                    [
                        "torch==2.2.0",
                        "transformers",
                        "numpy",
                        "opencv_python",
                        "torchmetrics",
                        "supervision",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(setup_runtime_module.subprocess, "run") as run_mock:
                filtered_path = install_univad_requirements_without_torch(requirements_path)

            filtered_text = filtered_path.read_text(encoding="utf-8")
            self.assertNotIn("torch==2.2.0", filtered_text)
            self.assertNotIn("transformers", filtered_text)
            self.assertNotIn("numpy", filtered_text)
            self.assertNotIn("opencv_python", filtered_text)
            self.assertIn("torchmetrics", filtered_text)
            self.assertIn("supervision", filtered_text)

            command = run_mock.call_args.args[0]
            self.assertIn("-c", command)
            constraints_path = Path(command[command.index("-c") + 1])
            constraints_text = constraints_path.read_text(encoding="utf-8")
            self.assertIn("transformers==4.45.2", constraints_text)
            self.assertIn("numpy==1.26.4", constraints_text)

    def test_runtime_constraints_pin_transformers_below_v5(self) -> None:
        constraints_text = write_univad_runtime_constraints().read_text(encoding="utf-8")

        self.assertIn("transformers==4.45.2", constraints_text)
        self.assertNotIn("transformers==5", constraints_text)

    def test_grounding_mask_generation_skips_complete_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            mask_root = root / "masks"
            univad_root = root / "UniVAD"
            config_root = univad_root / "configs" / "class_histogram"
            config_root.mkdir(parents=True)
            (univad_root / "models" / "GroundingDINO").mkdir(parents=True)
            for category in ("breakfast_box", "screw_bag"):
                (config_root / f"{category}.yaml").write_text(
                    "grounding_config:\n  text_prompt: object\n",
                    encoding="utf-8",
                )
                image_path = data_root / category / "train" / "good" / "000.png"
                image_path.parent.mkdir(parents=True)
                image_path.write_bytes(b"not-a-real-png")
            breakfast_mask = mask_root / "breakfast_box" / "train" / "good" / "000" / "grounding_mask.png"
            breakfast_mask.parent.mkdir(parents=True)
            breakfast_mask.write_bytes(b"mask")

            calls: list[tuple[list[str], str, dict[str, object]]] = []
            fake_component = types.ModuleType("models.component_segmentaion")
            fake_component.grounding_segmentation = lambda image_paths, output_dir, config: calls.append(
                (list(image_paths), output_dir, config)
            )
            fake_models = types.ModuleType("models")
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            fake_yaml = types.SimpleNamespace(safe_load=lambda _text: {"grounding_config": {"text_prompt": "object"}})
            previous_models = sys.modules.get("models")
            previous_component = sys.modules.get("models.component_segmentaion")
            previous_torch = sys.modules.get("torch")
            previous_yaml = sys.modules.get("yaml")
            try:
                sys.modules["models"] = fake_models
                sys.modules["models.component_segmentaion"] = fake_component
                sys.modules["torch"] = fake_torch
                sys.modules["yaml"] = fake_yaml
                spec = {
                    "data_root": data_root,
                    "mask_root": mask_root,
                    "external_dir": univad_root,
                    "groundingdino_dir": univad_root / "models" / "GroundingDINO",
                    "requires_cuda": False,
                    "required_data_paths": [],
                }

                self.assertEqual(
                    collect_univad_missing_mask_image_paths(spec, "breakfast_box"),
                    [],
                )
                self.assertEqual(
                    collect_univad_missing_mask_image_paths(spec, "screw_bag"),
                    [str(data_root / "screw_bag" / "train" / "good" / "000.png")],
                )

                with mock.patch.object(setup_runtime_module, "patch_univad_component_segmentation", return_value=False):
                    status = maybe_prepare_univad_grounding_masks(
                        spec,
                        categories=["breakfast_box", "screw_bag"],
                        settings={"auto_prepare_univad_grounding_masks": True},
                    )

                self.assertEqual(status["generated_categories"], ["screw_bag"])
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], [str(data_root / "screw_bag" / "train" / "good" / "000.png")])
                self.assertEqual(calls[0][1], str(mask_root / "screw_bag"))
            finally:
                if previous_models is None:
                    sys.modules.pop("models", None)
                else:
                    sys.modules["models"] = previous_models
                if previous_component is None:
                    sys.modules.pop("models.component_segmentaion", None)
                else:
                    sys.modules["models.component_segmentaion"] = previous_component
                if previous_torch is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = previous_torch
                if previous_yaml is None:
                    sys.modules.pop("yaml", None)
                else:
                    sys.modules["yaml"] = previous_yaml


if __name__ == "__main__":
    unittest.main()
