from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from augmentation_runtime import apply_augmentation, load_manifest


def sha256_image(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifest identity reproduction.")
    parser.add_argument(
        "--manifest",
        default="experiments/validation/condition_shift_baseline/manifests/query_identity.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/identity_reproduction_check.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means full manifest.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    entries = load_manifest(manifest_path)
    if args.limit > 0:
        entries = entries[: args.limit]

    mismatches: list[dict] = []
    compared = 0

    for entry in entries:
        source_path = Path(entry["source_path"])
        with Image.open(source_path) as image_obj:
            original = image_obj.convert("RGB")
            reproduced = apply_augmentation(
                original,
                augmentation_type=entry["augmentation_type"],
                severity=entry["severity"],
                seed=entry["seed"],
                params=entry["params"],
            )

        original_array = np.asarray(original, dtype=np.uint8)
        reproduced_array = np.asarray(reproduced, dtype=np.uint8)
        equal = np.array_equal(original_array, reproduced_array)
        compared += 1

        if not equal:
            mismatches.append(
                {
                    "source_path": str(source_path),
                    "source_id": entry["source_id"],
                    "augmentation_type": entry["augmentation_type"],
                    "severity": entry["severity"],
                    "original_sha256": sha256_image(original),
                    "reproduced_sha256": sha256_image(reproduced),
                }
            )

    report = {
        "manifest": str(manifest_path),
        "compared": compared,
        "mismatch_count": len(mismatches),
        "reproduced_exactly": len(mismatches) == 0,
        "mismatches": mismatches[:20],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
