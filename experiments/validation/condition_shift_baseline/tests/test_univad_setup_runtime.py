from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
for import_dir in (SRC_DIR, SRC_DIR / "core", SRC_DIR / "orchestration"):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from univad.setup_runtime import (  # noqa: E402
    checkpoint_download_urls,
    checkpoint_file_ready,
    collect_missing_checkpoint_files,
)
from univad.transformers_runtime import disable_transformers_tensorflow_backend  # noqa: E402


class UniVADSetupRuntimeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
