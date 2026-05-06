from __future__ import annotations

import argparse
import json
import sys
import time
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
from relation.sam_lad_components import SamLadComponentConfig, SamLadComponentModel  # noqa: E402


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
    component_model: str = "proxy",
    sam_checkpoint: Path | None = None,
    sam_root: Path | None = None,
    sam_device: str = "auto",
    sam_min_area_ratio: float = 0.0005,
    sam_small_area_ratio: float = 0.006,
    sam_max_area_ratio: float = 0.85,
    sam_points_per_side: int = 32,
    sam_crop_n_layers: int = 1,
    progress_every: int = 1,
) -> dict[str, Any]:
    entries = [
        entry
        for entry in load_manifest(manifest_path)
        if entry.get("category") == category
        and entry.get("augmentation_type") == "position_shift"
        and entry.get("severity") == severity
    ][:limit]

    extractor = build_component_extractor(
        repo_root=repo_root,
        component_model=component_model,
        max_components=max_components,
        sam_checkpoint=sam_checkpoint,
        sam_root=sam_root,
        sam_device=sam_device,
        sam_min_area_ratio=sam_min_area_ratio,
        sam_small_area_ratio=sam_small_area_ratio,
        sam_max_area_ratio=sam_max_area_ratio,
        sam_points_per_side=sam_points_per_side,
        sam_crop_n_layers=sam_crop_n_layers,
    )
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    run_started_at = time.perf_counter()
    for entry_index, entry in enumerate(entries, start=1):
        item_started_at = time.perf_counter()
        source_path = resolve_source_path(entry, repo_root)
        image = Image.open(source_path).convert("RGB")
        extraction = extractor(image)
        components = extraction["components"]
        elapsed = time.perf_counter() - item_started_at
        if progress_every > 0 and (entry_index == 1 or entry_index % progress_every == 0 or entry_index == len(entries)):
            print(
                "relation probe progress: "
                f"{entry_index}/{len(entries)} "
                f"component_model={component_model} "
                f"source_id={entry.get('source_id')} "
                f"raw_masks={extraction.get('raw_mask_count', 0)} "
                f"components={len(components)} "
                f"merged_small={extraction.get('merged_small_count', 0)} "
                f"item_sec={elapsed:.1f} total_sec={time.perf_counter() - run_started_at:.1f} "
                f"note={extraction.get('component_note', '-')}",
                flush=True,
            )
        if len(components) < 2:
            skipped.append(
                {
                    "source_id": entry.get("source_id"),
                    "reason": "fewer_than_two_components",
                    "component_model": component_model,
                    "component_source": extraction["component_source"],
                    "component_count": len(components),
                    "raw_mask_count": extraction.get("raw_mask_count", 0),
                    "merged_small_count": extraction.get("merged_small_count", 0),
                    "component_note": extraction.get("component_note", "-"),
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
                "component_model": component_model,
                "component_source": extraction["component_source"],
                "component_count": len(components),
                "raw_mask_count": extraction.get("raw_mask_count", 0),
                "merged_small_count": extraction.get("merged_small_count", 0),
                "component_note": extraction.get("component_note", "-"),
                "component_sources": ",".join(sorted({component.source for component in components})),
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
        "component_model": component_model,
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


def build_component_extractor(
    *,
    repo_root: Path,
    component_model: str,
    max_components: int,
    sam_checkpoint: Path | None,
    sam_root: Path | None,
    sam_device: str,
    sam_min_area_ratio: float,
    sam_small_area_ratio: float,
    sam_max_area_ratio: float,
    sam_points_per_side: int,
    sam_crop_n_layers: int,
):
    if component_model == "proxy":
        return lambda image: {
            "components": extract_proxy_components(image, max_components=max_components),
            "component_source": "proxy",
            "raw_mask_count": 0,
            "merged_small_count": 0,
            "component_note": "-",
        }
    if component_model != "sam_lad":
        raise ValueError(f"Unsupported component_model={component_model}")
    config = SamLadComponentConfig(
        max_components=max_components,
        min_area_ratio=sam_min_area_ratio,
        small_area_ratio=sam_small_area_ratio,
        max_area_ratio=sam_max_area_ratio,
        points_per_side=sam_points_per_side,
        crop_n_layers=sam_crop_n_layers,
    )
    model = SamLadComponentModel.from_repo(
        repo_root=repo_root,
        checkpoint_path=sam_checkpoint,
        sam_root=sam_root,
        device=sam_device,
        config=config,
    )

    def extract_with_sam_lad(image):
        result = model.extract(image)
        return {
            "components": result.components,
            "component_source": result.component_source,
            "raw_mask_count": result.raw_mask_count,
            "merged_small_count": result.merged_small_count,
            "component_note": result.note,
        }

    return extract_with_sam_lad


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
    parser.add_argument("--component-model", choices=["proxy", "sam_lad"], default="proxy")
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--sam-root", type=Path)
    parser.add_argument("--sam-device", default="auto")
    parser.add_argument("--sam-min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--sam-small-area-ratio", type=float, default=0.006)
    parser.add_argument("--sam-max-area-ratio", type=float, default=0.85)
    parser.add_argument("--sam-points-per-side", type=int, default=32)
    parser.add_argument("--sam-crop-n-layers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1)
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
        component_model=args.component_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_root=args.sam_root,
        sam_device=args.sam_device,
        sam_min_area_ratio=args.sam_min_area_ratio,
        sam_small_area_ratio=args.sam_small_area_ratio,
        sam_max_area_ratio=args.sam_max_area_ratio,
        sam_points_per_side=args.sam_points_per_side,
        sam_crop_n_layers=args.sam_crop_n_layers,
        progress_every=args.progress_every,
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "skipped"}}, indent=2))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
