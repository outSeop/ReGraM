from __future__ import annotations

from itertools import combinations
from statistics import mean, median
from typing import Any

from graph_probe.component_matching import MatchResult, match_components
from graph_probe.graph_features import edge_feature, edge_residual, node_geometry_residual


def graph_consistency_score(
    ref_nodes: list[dict[str, Any]],
    query_nodes: list[dict[str, Any]],
    *,
    image_size: tuple[int, int] | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    max_match_cost: float | None = None,
) -> dict[str, Any]:
    """Compute rule-based reference-query component graph consistency scores."""
    matching = match_components(
        ref_nodes,
        query_nodes,
        image_size=image_size,
        max_cost=max_match_cost,
    )
    node_residuals = [
        node_geometry_residual(ref_nodes[ref_idx], query_nodes[query_idx], image_size=image_size)
        for ref_idx, query_idx, _ in matching.matches
    ]
    edge_residuals = _matched_edge_residuals(ref_nodes, query_nodes, matching, image_size=image_size)
    s_node = _mean_node_residual(node_residuals)
    s_edge = _mean_edge_residual(edge_residuals)
    s_unmatched = float(len(matching.unmatched_ref) + len(matching.unmatched_query))
    s_total = alpha * s_node + beta * s_edge + gamma * s_unmatched
    return {
        "S_node": s_node,
        "S_edge": s_edge,
        "S_unmatched": s_unmatched,
        "S_total": float(s_total),
        "matched_node_count": len(matching.matches),
        "matched_node_ratio": _matched_ratio(len(matching.matches), len(ref_nodes), len(query_nodes)),
        "unmatched_ref_count": len(matching.unmatched_ref),
        "unmatched_query_count": len(matching.unmatched_query),
        "node_count_delta": int(len(query_nodes) - len(ref_nodes)),
        "mean_centroid_shift": _mean([item["centroid_shift"] for item in node_residuals]),
        "median_centroid_shift": _median([item["centroid_shift"] for item in node_residuals]),
        "mean_area_ratio_change": _mean([item["area_ratio_change"] for item in node_residuals]),
        "mean_bbox_iou": _mean([item["bbox_iou"] for item in node_residuals]),
        "mean_edge_distance_change": _mean([item["distance"] for item in edge_residuals]),
        "mean_edge_angle_change": _mean([item["angle"] for item in edge_residuals]),
        "matching": {
            "matches": [
                {"ref_index": ref_idx, "query_index": query_idx, "cost": cost}
                for ref_idx, query_idx, cost in matching.matches
            ],
            "unmatched_ref": matching.unmatched_ref,
            "unmatched_query": matching.unmatched_query,
        },
        "node_residuals": node_residuals,
        "edge_residuals": edge_residuals,
    }


def score_row(
    *,
    ref_nodes: list[dict[str, Any]],
    query_nodes: list[dict[str, Any]],
    category: str,
    condition: str,
    image_id: str,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Return a compact CSV-friendly graph consistency row."""
    score = graph_consistency_score(ref_nodes, query_nodes, image_size=image_size)
    return {
        "category": category,
        "condition": condition,
        "image_id": image_id,
        "node_delta": score["node_count_delta"],
        "matched_ratio": score["matched_node_ratio"],
        "S_node": score["S_node"],
        "S_edge": score["S_edge"],
        "S_unmatched": score["S_unmatched"],
        "S_total": score["S_total"],
        "mean_centroid_shift": score["mean_centroid_shift"],
        "mean_area_ratio_change": score["mean_area_ratio_change"],
        "mean_bbox_iou": score["mean_bbox_iou"],
        "mean_edge_distance_change": score["mean_edge_distance_change"],
        "mean_edge_angle_change": score["mean_edge_angle_change"],
    }


def _matched_edge_residuals(
    ref_nodes: list[dict[str, Any]],
    query_nodes: list[dict[str, Any]],
    matching: MatchResult,
    *,
    image_size: tuple[int, int] | None,
) -> list[dict[str, float]]:
    residuals: list[dict[str, float]] = []
    pairs = [(ref_idx, query_idx) for ref_idx, query_idx, _ in matching.matches]
    for (ref_a, query_a), (ref_b, query_b) in combinations(pairs, 2):
        ref_edge = edge_feature(ref_nodes[ref_a], ref_nodes[ref_b], image_size=image_size)
        query_edge = edge_feature(query_nodes[query_a], query_nodes[query_b], image_size=image_size)
        residuals.append(edge_residual(ref_edge, query_edge))
    return residuals


def _mean_node_residual(residuals: list[dict[str, float]]) -> float:
    if not residuals:
        return 0.0
    values = [
        item["centroid_shift"]
        + item["area_ratio_change"]
        + item["bbox_w_change"]
        + item["bbox_h_change"]
        + (1.0 - item["bbox_iou"])
        for item in residuals
    ]
    return _mean(values)


def _mean_edge_residual(residuals: list[dict[str, float]]) -> float:
    if not residuals:
        return 0.0
    values = [
        item["dx"]
        + item["dy"]
        + item["distance"]
        + item["angle"]
        + item["log_area_ratio"]
        for item in residuals
    ]
    return _mean(values)


def _matched_ratio(match_count: int, ref_count: int, query_count: int) -> float:
    denominator = max(ref_count, query_count, 1)
    return float(match_count / denominator)


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0
