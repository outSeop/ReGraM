"""Build query manifests for condition-shift experiments.

Role:
- manifest builder
- input: query image root grouped by category
- output: jsonl manifest consumed by evaluation runners

Used by:
- notebook runner setup
- manual CLI preprocessing before PatchCore shift evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.augmentation_runtime import build_manifest_entries, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query manifest for on-the-fly augmentations.")
    parser.add_argument("--input-root", default="data/query_normal_clean")
    parser.add_argument(
        "--output",
        default="manifests/query_identity.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=None,
        help="Override augmentation kinds. Ignored when --identity-only is set.",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        default=None,
        help="Override severities. Ignored when --identity-only is set.",
    )
    args = parser.parse_args()

    entries = build_manifest_entries(
        Path(args.input_root),
        augmentations=args.augmentations,
        severities=args.severities,
        identity_only=args.identity_only,
        seed=args.seed,
    )
    output_path = Path(args.output)
    write_manifest(entries, output_path)

    print(f"manifest={output_path}")
    print(f"entries={len(entries)}")


if __name__ == "__main__":
    main()
