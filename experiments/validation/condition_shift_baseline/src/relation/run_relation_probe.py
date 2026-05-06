from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data.augmentation_runtime import apply_augmentation, load_manifest  # noqa: E402
from relation.component_extraction import component_centroids, extract_proxy_components  # noqa: E402
from relation.geometry import (  # noqa: E402
    apply_position_transform,
    relation_score_bundle,
    resolve_position_shift_transform,
)


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "index.md").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from: {start}")


def resolve_source_path(entry: dict[str, Any], repo_root: Path) -> Path:
    path = Path(entry["source_path"])
    if entry.get("source_path_mode") == "repo_relative" or not path.is_absolute():
        return repo_root / path
    return path


def run_probe(
    *,
    repo_root: Path,
    manifest_path: Path,
    category: str,
    severity: str,
    limit: int,
    output_path: Path,
    max_components: int,
) -> dict[str, Any]:
    entries = [
        entry
        for entry in load_manifest(manifest_path)
        if entry.get("category") == category
        and entry.get("augmentation_type") == "position_shift"
        and entry.get("severity") == severity
    ][:limit]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in entries:
        source_path = resolve_source_path(entry, repo_root)
        image = Image.open(source_path).convert("RGB")
        components = extract_proxy_components(image, max_components=max_components)
        if len(components) < 2:
            skipped.append(
                {
                    "source_id": entry.get("source_id"),
                    "reason": "fewer_than_two_proxy_components",
                    "component_count": len(components),
                }
            )
            continue

        reference_points = component_centroids(components)
        transform = resolve_position_shift_transform(image.size, entry.get("params", {}), int(entry["seed"]))
        transformed_points = apply_position_transform(reference_points, transform)
        scores = relation_score_bundle(reference_points, transformed_points)

        shifted_image = apply_augmentation(
            image,
            augmentation_type="position_shift",
            severity=severity,
            seed=int(entry["seed"]),
            params=entry.get("params"),
        )
        rows.append(
            {
                "source_id": entry.get("source_id"),
                "source_path": str(source_path),
                "component_count": len(components),
                "placement": transform.placement,
                "scale": transform.scale,
                "offset_x": transform.offset[0],
                "offset_y": transform.offset[1],
                "image_width": image.size[0],
                "image_height": image.size[1],
                "shifted_width": shifted_image.size[0],
                "shifted_height": shifted_image.size[1],
                **scores,
            }
        )

    summary = {
        "category": category,
        "severity": severity,
        "manifest_path": str(manifest_path),
        "requested_limit": limit,
        "evaluated_count": len(rows),
        "skipped_count": len(skipped),
        "metric_medians": _metric_medians(rows),
        "rows": rows,
        "skipped": skipped,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _metric_medians(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    metrics = ["s_abs", "s_centered_raw", "s_centered_norm", "s_pair_raw", "s_pair_norm"]
    medians: dict[str, float | None] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        medians[metric] = float(median(values)) if values else None
    return medians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=find_repo_root(Path.cwd()))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/query_position_shift.jsonl"))
    parser.add_argument("--category", default="breakfast_box")
    parser.add_argument("--severity", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/validation/condition_shift_baseline/reports/relation_probe/position_shift_known_transform.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    summary = run_probe(
        repo_root=repo_root,
        manifest_path=manifest_path,
        category=args.category,
        severity=args.severity,
        limit=args.limit,
        output_path=output_path,
        max_components=args.max_components,
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "skipped"}}, indent=2))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
