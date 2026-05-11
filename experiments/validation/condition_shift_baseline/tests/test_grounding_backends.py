from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from univad.grounding_backends import grounding_segmentation_with_backend  # noqa: E402


class GroundingBackendTests(unittest.TestCase):
    def test_rejects_unknown_segmentation_backend_without_external_imports(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported segmentation backend"):
            grounding_segmentation_with_backend(
                [],
                "/tmp/out",
                {},
                backend="unknown",
            )

    def test_robustsam_backend_requires_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "robustsam_root and robustsam_checkpoint"):
            grounding_segmentation_with_backend(
                [],
                "/tmp/out",
                {},
                backend="robustsam",
            )


if __name__ == "__main__":
    unittest.main()
