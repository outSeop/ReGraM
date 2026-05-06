from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from graph_probe.component_io import component_quality_row, iter_summary_rows, save_csv, save_json  # noqa: E402
from graph_probe.graph_consistency import graph_consistency_score, score_row  # noqa: E402


def run_logical_probe(
    *,
    reference_path: Path,
    normal_query_path: Path | None,
    logical_query_path: Path,
    category: str,
    output_json: Path,
    output_csv: Path | None = None,
    quality_csv: Path | None = None,
) -> dict[str, Any]:
    """Compare reference graphs against normal and logical-anomaly queries."""
    reference_rows = iter_summary_rows(reference_path)
    reference_by_id = _rows_by_source_id(reference_rows)
    score_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    inputs = []
    if normal_query_path is not None:
        inputs.append(("normal_query", normal_query_path))
    inputs.append(("logical_anomaly", logical_query_path))

    for condition, query_path in inputs:
        for query_row in iter_summary_rows(query_path):
            image_id = str(query_row.get("source_id", query_row.get("image_id", query_path.stem)))
            ref_row = reference_by_id.get(image_id, reference_rows[0])
            ref_nodes = list(ref_row.get("component_nodes", []))
            query_nodes = list(query_row.get("component_nodes", []))
            image_size = _image_size_from_rows(ref_row, query_row)
            compact = score_row(
                ref_nodes=ref_nodes,
                query_nodes=query_nodes,
                category=category,
                condition=condition,
                image_id=image_id,
                image_size=image_size,
            )
            detail = graph_consistency_score(ref_nodes, query_nodes, image_size=image_size)
            score_rows.append(compact)
            details.append(
                {
                    **compact,
                    "reference_path": str(reference_path),
                    "query_path": str(query_path),
                    "detail": detail,
                }
            )
            quality_rows.append(
                component_quality_row(
                    nodes=query_nodes,
                    image_id=image_id,
                    category=category,
                    split="query",
                    shift_type=condition,
                    severity="none",
                    num_raw_masks=_optional_int(query_row.get("raw_mask_count")),
                )
            )

    result = {
        "probe": "logical",
        "category": category,
        "reference_path": str(reference_path),
        "normal_query_path": str(normal_query_path) if normal_query_path is not None else None,
        "logical_query_path": str(logical_query_path),
        "rows": score_rows,
        "details": details,
        "quality_rows": quality_rows,
    }
    save_json(result, output_json)
    if output_csv is not None:
        save_csv(score_rows, output_csv)
    if quality_csv is not None:
        save_csv(quality_rows, quality_csv)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--normal-query", type=Path)
    parser.add_argument("--logical-query", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--quality-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_logical_probe(
        reference_path=args.reference,
        normal_query_path=args.normal_query,
        logical_query_path=args.logical_query,
        category=args.category,
        output_json=args.output_json,
        output_csv=args.output_csv,
        quality_csv=args.quality_csv,
    )
    print(f"rows={len(result['rows'])}")
    print(f"output_json={args.output_json}")


def _rows_by_source_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("source_id", row.get("image_id"))): row
        for row in rows
        if row.get("source_id", row.get("image_id")) is not None
    }


def _image_size_from_rows(*rows: dict[str, Any]) -> tuple[int, int] | None:
    for row in rows:
        width = row.get("image_width") or row.get("width")
        height = row.get("image_height") or row.get("height")
        if width and height:
            return int(width), int(height)
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()
