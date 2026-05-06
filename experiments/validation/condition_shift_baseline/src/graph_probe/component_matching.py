from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from graph_probe.graph_features import node_feature


@dataclass(frozen=True)
class MatchResult:
    matches: list[tuple[int, int, float]]
    unmatched_ref: list[int]
    unmatched_query: list[int]
    cost_matrix: list[list[float]]


def match_components(
    ref_nodes: list[dict[str, Any]],
    query_nodes: list[dict[str, Any]],
    *,
    image_size: tuple[int, int] | None = None,
    max_cost: float | None = None,
    weights: dict[str, float] | None = None,
) -> MatchResult:
    """Match reference and query component nodes with a deterministic assignment."""
    if not ref_nodes or not query_nodes:
        return MatchResult(
            matches=[],
            unmatched_ref=list(range(len(ref_nodes))),
            unmatched_query=list(range(len(query_nodes))),
            cost_matrix=[],
        )
    cost_matrix = build_cost_matrix(
        ref_nodes,
        query_nodes,
        image_size=image_size,
        weights=weights,
    )
    assignments = _linear_sum_assignment(cost_matrix)
    matches = [
        (ref_idx, query_idx, float(cost_matrix[ref_idx, query_idx]))
        for ref_idx, query_idx in assignments
        if max_cost is None or float(cost_matrix[ref_idx, query_idx]) <= max_cost
    ]
    matched_ref = {ref_idx for ref_idx, _, _ in matches}
    matched_query = {query_idx for _, query_idx, _ in matches}
    return MatchResult(
        matches=matches,
        unmatched_ref=[index for index in range(len(ref_nodes)) if index not in matched_ref],
        unmatched_query=[index for index in range(len(query_nodes)) if index not in matched_query],
        cost_matrix=cost_matrix.tolist(),
    )


def build_cost_matrix(
    ref_nodes: list[dict[str, Any]],
    query_nodes: list[dict[str, Any]],
    *,
    image_size: tuple[int, int] | None = None,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Build geometry-first node matching costs."""
    resolved_weights = {
        "centroid": 1.0,
        "area": 1.0,
        "bbox": 0.5,
        **(weights or {}),
    }
    costs = np.zeros((len(ref_nodes), len(query_nodes)), dtype=np.float64)
    for ref_idx, ref_node in enumerate(ref_nodes):
        ref_feat = node_feature(ref_node, image_size=image_size)
        for query_idx, query_node in enumerate(query_nodes):
            query_feat = node_feature(query_node, image_size=image_size)
            centroid = (
                (ref_feat["cx"] - query_feat["cx"]) ** 2
                + (ref_feat["cy"] - query_feat["cy"]) ** 2
            ) ** 0.5
            area = abs(ref_feat["area_ratio"] - query_feat["area_ratio"])
            bbox = abs(ref_feat["bbox_w"] - query_feat["bbox_w"]) + abs(ref_feat["bbox_h"] - query_feat["bbox_h"])
            costs[ref_idx, query_idx] = (
                resolved_weights["centroid"] * centroid
                + resolved_weights["area"] * area
                + resolved_weights["bbox"] * bbox
            )
    return costs


def _linear_sum_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: WPS433

        rows, cols = linear_sum_assignment(cost_matrix)
        return list(zip([int(row) for row in rows], [int(col) for col in cols]))
    except Exception:  # noqa: BLE001
        return _dp_assignment(cost_matrix)


def _dp_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = cost_matrix.shape
    transposed = False
    matrix = cost_matrix
    if rows > cols:
        matrix = cost_matrix.T
        rows, cols = matrix.shape
        transposed = True
    if cols > 20:
        return _greedy_assignment(cost_matrix)

    @lru_cache(maxsize=None)
    def solve(row: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if row == rows:
            return 0.0, ()
        best_cost = float("inf")
        best_pairs: tuple[tuple[int, int], ...] = ()
        for col in range(cols):
            if used_mask & (1 << col):
                continue
            tail_cost, tail_pairs = solve(row + 1, used_mask | (1 << col))
            total = float(matrix[row, col]) + tail_cost
            if total < best_cost:
                best_cost = total
                best_pairs = ((row, col), *tail_pairs)
        return best_cost, best_pairs

    _, pairs = solve(0, 0)
    if transposed:
        return [(col, row) for row, col in pairs]
    return list(pairs)


def _greedy_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    candidates = [
        (float(cost_matrix[row, col]), row, col)
        for row in range(cost_matrix.shape[0])
        for col in range(cost_matrix.shape[1])
    ]
    matches: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _, row, col in sorted(candidates):
        if row in used_rows or col in used_cols:
            continue
        matches.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    return matches

