from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data.augmentation_runtime import apply_augmentation, load_manifest
from graph_probe.component_io import save_json
from relation.grounding_mask_cluster import cluster_masks_to_components, raw_masks_from_label_image


def build_grounding_mask_cluster_summary(
    *,
    repo_root: Path,
    manifest_path: Path,
    category: str,
    augmentation_type: str,
    severity: str,
    limit: int,
    mask_root: Path,
    mask_data_root: Path,
    cluster_config: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build component-node summaries for clean or shifted normal manifest rows.

    The segmentation input is an existing grounding mask. The image can be
    augmented to measure graph-score stability under appearance shifts without
    invoking SAM or GroundingDINO in this probe.
    """
    entries = _selected_manifest_entries(
        manifest_path=manifest_path,
        category=category,
        augmentation_type=augmentation_type,
        severity=severity,
        limit=limit,
    )
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in entries:
        source_path = _resolve_source_path(entry, repo_root=repo_root)
        mask_path = expected_grounding_mask_path_for_image(
            source_path,
            data_root=mask_data_root,
            mask_root=mask_root,
            category=category,
        )
        if not mask_path.exists():
            skipped.append(
                {
                    "base_image_id": entry.get("source_id"),
                    "source_id": entry.get("source_id"),
                    "reason": "grounding_mask_missing",
                    "mask_path": str(mask_path),
                }
            )
            continue

        image = Image.open(source_path).convert("RGB")
        query_image = apply_augmentation(
            image,
            augmentation_type=augmentation_type,
            severity=severity,
            seed=int(entry.get("seed", 0)),
            params=entry.get("params"),
        )
        with Image.open(mask_path) as mask_image:
            raw_masks = raw_masks_from_label_image(mask_image)
        nodes = cluster_masks_to_components(
            np.asarray(query_image.convert("RGB")),
            raw_masks,
            config=cluster_config,
            category=category,
        )
        type_counts = _node_type_counts(nodes)
        rows.append(
            {
                "base_image_id": entry.get("base_image_id", entry.get("source_id")),
                "source_id": entry.get("source_id"),
                "image_id": entry.get("source_id"),
                "source_path": str(source_path),
                "category": category,
                "augmentation_type": augmentation_type,
                "severity": severity,
                "component_model": "grounding_mask_cluster",
                "component_source": "grounding_mask_cluster_existing_mask",
                "component_count": len(nodes),
                "raw_mask_count": len(raw_masks),
                "thing_count": type_counts.get("thing", 0),
                "stuff_cluster_count": type_counts.get("stuff_cluster", 0),
                "small_isolated_count": type_counts.get("small_isolated", 0),
                "image_width": query_image.size[0],
                "image_height": query_image.size[1],
                "mask_path": str(mask_path),
                "component_nodes": nodes,
            }
        )

    summary = {
        "probe": "component_summary",
        "category": category,
        "augmentation_type": augmentation_type,
        "severity": severity,
        "component_model": "grounding_mask_cluster",
        "manifest_path": str(manifest_path),
        "requested_limit": limit,
        "evaluated_count": len(rows),
        "skipped_count": len(skipped),
        "rows": rows,
        "skipped": skipped,
    }
    if output_path is not None:
        save_json(summary, output_path)
    return summary


def load_config_file(path: Path | None) -> dict[str, Any] | None:
    """Load a YAML/JSON config file when provided."""
    if path is None:
        return None
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(f"PyYAML is required to load {path}") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return json.loads(path.read_text(encoding="utf-8"))


def expected_grounding_mask_path_for_image(
    image_path: Path,
    *,
    data_root: Path,
    mask_root: Path,
    category: str,
) -> Path:
    """Resolve the existing UniVAD grounding-mask path for a source image."""
    try:
        rel_path = image_path.relative_to(data_root / category)
    except ValueError:
        parts = list(image_path.parts)
        if category not in parts:
            raise
        category_index = len(parts) - 1 - parts[::-1].index(category)
        rel_path = Path(*parts[category_index + 1 :])
    return mask_root / category / rel_path.with_suffix("") / "grounding_mask.png"


def _selected_manifest_entries(
    *,
    manifest_path: Path,
    category: str,
    augmentation_type: str,
    severity: str,
    limit: int,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in load_manifest(manifest_path)
        if entry.get("category") == category
        and entry.get("augmentation_type") == augmentation_type
        and entry.get("severity") == severity
    ]
    return entries[:limit]


def _resolve_source_path(entry: dict[str, Any], *, repo_root: Path) -> Path:
    path = Path(entry["source_path"])
    if entry.get("source_path_mode") == "repo_relative" or not path.is_absolute():
        return repo_root / path
    return path


def _node_type_counts(nodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        node_type = str(node["node_type"])
        counts[node_type] = counts.get(node_type, 0) + 1
    return counts
