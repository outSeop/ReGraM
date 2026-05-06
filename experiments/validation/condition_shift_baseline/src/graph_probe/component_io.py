from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Load a JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: Any, path: Path) -> None:
    """Save JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Save row dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_component_nodes(path: Path, *, row_index: int | None = None) -> list[dict[str, Any]]:
    """Load a component node list or one row's component_nodes from a summary JSON."""
    document = load_json(path)
    if isinstance(document, list):
        return document
    if not isinstance(document, dict):
        raise ValueError(f"Unsupported component JSON shape: {path}")
    if "component_nodes" in document:
        return list(document["component_nodes"])
    if "nodes" in document:
        return list(document["nodes"])
    rows = document.get("rows")
    if isinstance(rows, list):
        if row_index is None:
            if len(rows) != 1:
                raise ValueError(
                    f"row_index is required when summary has {len(rows)} rows: {path}"
                )
            row_index = 0
        return list(rows[row_index].get("component_nodes", []))
    raise ValueError(f"No component nodes found in: {path}")


def iter_summary_rows(path: Path) -> list[dict[str, Any]]:
    """Return rows from a summary JSON, or wrap a bare component-node list as one row."""
    document = load_json(path)
    if isinstance(document, dict) and isinstance(document.get("rows"), list):
        return list(document["rows"])
    nodes = load_component_nodes(path)
    return [{"source_id": path.stem, "component_nodes": nodes}]


def component_quality_row(
    *,
    nodes: list[dict[str, Any]],
    image_id: str,
    category: str,
    split: str,
    shift_type: str,
    severity: str,
    num_raw_masks: int | None = None,
) -> dict[str, Any]:
    """Summarize component-node quality and composition for one image."""
    areas = [float(node.get("area", 0.0)) for node in nodes]
    node_types = [str(node.get("node_type", "")) for node in nodes]
    thing_area_sum = _area_sum(nodes, "thing")
    stuff_area_sum = _area_sum(nodes, "stuff_cluster")
    small_area_sum = _area_sum(nodes, "small_isolated")
    return {
        "image_id": image_id,
        "category": category,
        "split": split,
        "shift_type": shift_type,
        "severity": severity,
        "num_raw_masks": int(num_raw_masks if num_raw_masks is not None else _num_raw_masks(nodes)),
        "num_thing": node_types.count("thing"),
        "num_stuff_cluster": node_types.count("stuff_cluster"),
        "num_small_isolated": node_types.count("small_isolated"),
        "num_total_nodes": len(nodes),
        "thing_area_sum": thing_area_sum,
        "stuff_area_sum": stuff_area_sum,
        "small_area_sum": small_area_sum,
        "mean_node_area": float(sum(areas) / len(areas)) if areas else 0.0,
        "median_node_area": _median(areas),
    }


def _area_sum(nodes: list[dict[str, Any]], node_type: str) -> float:
    return float(sum(float(node.get("area", 0.0)) for node in nodes if node.get("node_type") == node_type))


def _num_raw_masks(nodes: list[dict[str, Any]]) -> int:
    mask_ids = {str(mask_id) for node in nodes for mask_id in node.get("mask_ids", [])}
    return len(mask_ids)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[middle])
    return float((sorted_values[middle - 1] + sorted_values[middle]) / 2.0)

