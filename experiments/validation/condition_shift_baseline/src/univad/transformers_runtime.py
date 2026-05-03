"""Runtime guards for UniVAD/GroundingDINO's Transformers usage."""

from __future__ import annotations

import os
import sys


def disable_transformers_tensorflow_backend() -> None:
    """Keep Transformers on the PyTorch path even when TensorFlow is installed.

    Colab images often include TensorFlow, but UniVAD/GroundingDINO only needs
    Transformers tokenization/model utilities through PyTorch. If TensorFlow has
    a protobuf mismatch, Transformers' tensor-conversion probe can fail during
    tokenizer.decode unless the TensorFlow backend is disabled.
    """
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    import_utils = sys.modules.get("transformers.utils.import_utils")
    if import_utils is not None:
        setattr(import_utils, "_tf_available", False)
        setattr(import_utils, "_tf_version", "N/A")

    for module_name in ("transformers.utils", "transformers.utils.generic"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "is_tf_available", lambda: False)
