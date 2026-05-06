from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from graph_probe.component_io import iter_summary_rows
from graph_probe.graph_consistency import score_row


SCORE_COLUMNS = [
    "S_node",
    "S_edge",
    "S_unmatched",
    "S_total",
    "matched_ratio",
    "node_delta",
    "mean_edge_distance_change",
    "mean_edge_angle_change",
]


def compare_summary_conditions(
    reference_summary_path: Path,
    query_summary_paths: dict[str, Path],
    category: str,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Compare a clean/reference summary against multiple condition summaries.

    Returns one row per matched reference-query image. Matching uses row_identity:
    base_image_id, source_id, image_id, then source_path.name.
    """
    reference_rows = iter_summary_rows(Path(reference_summary_path))
    reference_by_id = _rows_by_identity(reference_rows)
    rows: list[dict[str, Any]] = []

    for condition, query_summary_path in query_summary_paths.items():
        query_path = Path(query_summary_path)
        query_rows = iter_summary_rows(query_path)
        query_by_id = _rows_by_identity(query_rows)
        for image_id, reference_row in reference_by_id.items():
            query_row = query_by_id.get(image_id)
            if query_row is None:
                continue
            image_size = _image_size_from_rows(reference_row, query_row)
            row = score_row(
                ref_nodes=list(reference_row.get("component_nodes", [])),
                query_nodes=list(query_row.get("component_nodes", [])),
                category=category,
                condition=condition,
                image_id=image_id,
                image_size=image_size,
            )
            row.update(
                {
                    "reference_image_id": row_identity(reference_row),
                    "query_image_id": row_identity(query_row),
                    "reference_source_id": reference_row.get("source_id"),
                    "query_source_id": query_row.get("source_id"),
                    "reference_summary_path": str(reference_summary_path),
                    "query_summary_path": str(query_path),
                }
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=_score_table_columns())
    else:
        frame = frame[_score_table_columns()]
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
    return frame


def summarize_condition_scores(
    score_df: pd.DataFrame,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Summarize graph consistency scores by condition."""
    if score_df.empty:
        summary = pd.DataFrame(columns=["condition", "num_pairs"])
    else:
        grouped = score_df.groupby("condition", dropna=False)
        summary = grouped[SCORE_COLUMNS].agg(["mean", "median", "std", "min", "max"]).reset_index()
        summary.columns = [
            "_".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
        summary.insert(1, "num_pairs", grouped.size().to_numpy())
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False)
    return summary


def row_identity(row: dict[str, Any]) -> str | None:
    """Return the stable identity used for clean/shift/logical row matching."""
    for key in ("base_image_id", "source_id", "image_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    source_path = row.get("source_path")
    if source_path not in (None, ""):
        return Path(str(source_path)).name
    return None


def _rows_by_identity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row_identity(row)
        if identity is None:
            continue
        keyed.setdefault(identity, row)
    return keyed


def _image_size_from_rows(*rows: dict[str, Any]) -> tuple[int, int] | None:
    for row in rows:
        width = row.get("image_width") or row.get("width")
        height = row.get("image_height") or row.get("height")
        if width and height:
            return int(width), int(height)
    return None


def _score_table_columns() -> list[str]:
    return [
        "category",
        "condition",
        "image_id",
        "reference_image_id",
        "query_image_id",
        "reference_source_id",
        "query_source_id",
        "node_delta",
        "matched_ratio",
        "S_node",
        "S_edge",
        "S_unmatched",
        "S_total",
        "mean_centroid_shift",
        "mean_area_ratio_change",
        "mean_bbox_iou",
        "mean_edge_distance_change",
        "mean_edge_angle_change",
        "reference_summary_path",
        "query_summary_path",
    ]
