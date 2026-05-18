"""Normal-role graph prior probe for Stage 1 component candidates.

This module does not implement a model or train an anomaly detector. It turns
clean-normal component candidates into a role-level reference graph, then
checks whether query candidate pools can recover those normal roles under
different matching cost ablations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import atan2, log, pi, sqrt
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RolePrior:
    """Clean-normal statistics for one recurring component role."""

    role_id: str
    member_count: int
    occurrence_rate: float
    centroid_mean: list[float]
    centroid_std: list[float]
    area_ratio_mean: float
    area_ratio_std: float
    bbox_mean: list[float]
    reliability_mean: float
    dominant_normalization_type: str
    debug: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoleGraphPrior:
    """Reference graph built from clean-normal role priors."""

    category: str
    roles: list[RolePrior]
    edge_features: dict[str, dict[str, float]]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "roles": [role.to_dict() for role in self.roles],
            "edge_features": self.edge_features,
            "config": self.config,
        }


@dataclass(frozen=True)
class RoleMatchingConfig:
    """Cost and candidate-pool configuration for role-prior matching."""

    pool_mode: str = "loose"
    matching_mode: str = "geometry_dino_local_edge"
    max_match_cost: float = 1.20
    min_occurrence_rate: float = 0.15
    min_role_members: int = 2
    max_roles: int | None = None
    centroid_weight: float = 1.0
    area_weight: float = 0.7
    bbox_weight: float = 0.35
    type_weight: float = 0.25
    dino_weight: float = 0.45
    local_weight: float = 0.25
    edge_weight: float = 0.35
    unmatched_weight: float = 1.0
    count_weight: float = 0.2
    eps: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_role_graph_prior(
    clean_scores: list[dict[str, Any]],
    *,
    category: str,
    image_count: int,
    config: RoleMatchingConfig | None = None,
) -> RoleGraphPrior:
    """Estimate recurring normal roles from clean-normal candidate scores."""
    cfg = config or RoleMatchingConfig()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in clean_scores:
        if not _candidate_allowed(row, pool_mode="loose"):
            continue
        role_id = str(row.get("best_prototype_id") or "")
        if not role_id or role_id.lower() == "none":
            continue
        grouped.setdefault(role_id, []).append(row)

    roles: list[RolePrior] = []
    for role_id, rows in grouped.items():
        occurrence_rate = len({str(row.get("base_image_id", "")) for row in rows}) / max(float(image_count), 1.0)
        if len(rows) < cfg.min_role_members or occurrence_rate < cfg.min_occurrence_rate:
            continue
        centroids = np.asarray([_xy(row.get("centroid", [0.0, 0.0])) for row in rows], dtype=np.float64)
        areas = np.asarray([float(row.get("area_ratio", 0.0)) for row in rows], dtype=np.float64)
        bboxes = np.asarray([_bbox_norm(row.get("bbox", [0, 0, 0, 0])) for row in rows], dtype=np.float64)
        reliabilities = np.asarray([float(row.get("reliability", 0.0)) for row in rows], dtype=np.float64)
        roles.append(
            RolePrior(
                role_id=role_id,
                member_count=len(rows),
                occurrence_rate=float(occurrence_rate),
                centroid_mean=centroids.mean(axis=0).astype(float).tolist(),
                centroid_std=centroids.std(axis=0).astype(float).tolist(),
                area_ratio_mean=float(areas.mean()),
                area_ratio_std=float(areas.std()),
                bbox_mean=bboxes.mean(axis=0).astype(float).tolist(),
                reliability_mean=float(reliabilities.mean()) if len(reliabilities) else 0.0,
                dominant_normalization_type=_mode(str(row.get("normalization_type", "unknown")) for row in rows),
                debug={
                    "member_candidate_ids": [str(row.get("candidate_id")) for row in rows],
                    "member_image_ids": sorted({str(row.get("base_image_id", "")) for row in rows}),
                },
            )
        )
    roles.sort(key=lambda role: (-role.occurrence_rate, -role.member_count, role.role_id))
    if cfg.max_roles is not None:
        roles = roles[: max(0, int(cfg.max_roles))]
    return RoleGraphPrior(
        category=category,
        roles=roles,
        edge_features=_role_edge_features(roles),
        config=cfg.to_dict(),
    )


def score_query_against_role_prior(
    *,
    prior: RoleGraphPrior,
    query_scores: list[dict[str, Any]],
    category: str,
    condition: str,
    base_image_id: str,
    split: str,
    config: RoleMatchingConfig | None = None,
    patch_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Match one query candidate pool to the clean-normal role graph prior."""
    cfg = config or RoleMatchingConfig()
    candidates = [row for row in query_scores if _candidate_allowed(row, pool_mode=cfg.pool_mode)]
    cost_matrix = _role_candidate_cost_matrix(prior.roles, candidates, cfg, patch_summary=patch_summary)
    assignments = _linear_sum_assignment(cost_matrix) if len(prior.roles) and len(candidates) else []
    matches = [
        (role_index, candidate_index, float(cost_matrix[role_index, candidate_index]))
        for role_index, candidate_index in assignments
        if float(cost_matrix[role_index, candidate_index]) <= cfg.max_match_cost
    ]
    matched_roles = {role_index for role_index, _, _ in matches}
    matched_candidates = {candidate_index for _, candidate_index, _ in matches}
    unmatched_roles = [index for index in range(len(prior.roles)) if index not in matched_roles]
    unmatched_candidates = [index for index in range(len(candidates)) if index not in matched_candidates]
    node_residuals = [
        _role_node_residual(prior.roles[role_index], candidates[candidate_index])
        for role_index, candidate_index, _ in matches
    ]
    edge_residuals = _matched_edge_residuals(prior, candidates, matches)
    s_node = _mean(
        [
            item["centroid"]
            + item["area"]
            + item["bbox"]
            + item["type"]
            + item["dino"]
            + item["local"]
            for item in node_residuals
        ]
    )
    s_edge = _mean([item["residual"] for item in edge_residuals])
    s_unmatched = float(len(unmatched_roles) + len(unmatched_candidates))
    s_count = float(abs(len(candidates) - len(prior.roles)))
    s_total = s_node + cfg.edge_weight * s_edge + cfg.unmatched_weight * s_unmatched + cfg.count_weight * s_count
    role_count = max(len(prior.roles), 1)
    candidate_count = max(len(candidates), 1)
    matched_ratio = len(matches) / max(len(prior.roles), len(candidates), 1)
    role_coverage = len(matched_roles) / role_count
    dummy_match_rate = len(unmatched_roles) / role_count
    return {
        "category": category,
        "condition": condition,
        "base_image_id": base_image_id,
        "split": split,
        "pool_mode": cfg.pool_mode,
        "matching_mode": cfg.matching_mode,
        "num_reference_roles": len(prior.roles),
        "num_query_candidates": len(candidates),
        "matched_count": len(matches),
        "matched_ratio": float(matched_ratio),
        "role_coverage": float(role_coverage),
        "dummy_match_rate": float(dummy_match_rate),
        "unmatched_role_count": len(unmatched_roles),
        "unmatched_query_count": len(unmatched_candidates),
        "candidate_count": len(candidates),
        "fragment_burden": _fragment_burden(candidates),
        "large_stuff_role_retention": _large_stuff_retention(prior.roles, matched_roles),
        "stuff_match_rate": _stuff_match_rate(prior.roles, matched_roles),
        "S_node": float(s_node),
        "S_edge": float(s_edge),
        "S_unmatched": float(s_unmatched),
        "S_count": float(s_count),
        "S_total": float(s_total),
        "mean_match_cost": _mean([cost for _, _, cost in matches]),
        "matches": [
            {
                "role_id": prior.roles[role_index].role_id,
                "candidate_id": str(candidates[candidate_index].get("candidate_id")),
                "cost": cost,
                "node_type": str(candidates[candidate_index].get("node_type")),
                "normalization_type": str(candidates[candidate_index].get("normalization_type")),
            }
            for role_index, candidate_index, cost in matches
        ],
        "unmatched_roles": [prior.roles[index].role_id for index in unmatched_roles],
    }


def score_condition_table(
    *,
    prior: RoleGraphPrior,
    score_rows: list[dict[str, Any]],
    category: str,
    configs: list[RoleMatchingConfig],
    patch_summary_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score every condition/image candidate pool with every matching config."""
    patch_by_key = {
        (
            str(row.get("condition")),
            str(row.get("base_image_id")),
            str(row.get("target_type")),
            str(row.get("mask_source")),
            str(row.get("smoothing_mode")),
            float(row.get("lambda", 0.0)),
        ): row
        for row in (patch_summary_rows or [])
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in score_rows:
        key = (str(row.get("condition")), str(row.get("base_image_id")), str(row.get("split", "")))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for cfg in configs:
        for (condition, base_image_id, split), rows in sorted(grouped.items()):
            patch_summary = _patch_summary_for_mode(patch_by_key, condition, base_image_id, cfg.matching_mode)
            output.append(
                score_query_against_role_prior(
                    prior=prior,
                    query_scores=rows,
                    category=category,
                    condition=condition,
                    base_image_id=base_image_id,
                    split=split,
                    config=cfg,
                    patch_summary=patch_summary,
                )
            )
    return output


def compact_score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop verbose match lists before saving CSV tables."""
    return {
        key: value
        for key, value in row.items()
        if key not in {"matches", "unmatched_roles"}
    }


def _candidate_allowed(row: dict[str, Any], *, pool_mode: str) -> bool:
    node_type = str(row.get("node_type", ""))
    normalization_type = str(row.get("normalization_type", ""))
    if node_type in {"invalid_supermask", "invalid_fragment"}:
        return False
    if pool_mode == "strict":
        return bool(row.get("use_as_component_node", False)) and node_type == "valid_component"
    if pool_mode == "loose":
        return bool(row.get("use_for_patch_graph", True)) and normalization_type != "raw_candidate"
    raise ValueError(f"unsupported pool_mode: {pool_mode}")


def _role_candidate_cost_matrix(
    roles: list[RolePrior],
    candidates: list[dict[str, Any]],
    cfg: RoleMatchingConfig,
    *,
    patch_summary: dict[str, Any] | None,
) -> np.ndarray:
    matrix = np.zeros((len(roles), len(candidates)), dtype=np.float64)
    for role_index, role in enumerate(roles):
        for candidate_index, candidate in enumerate(candidates):
            residual = _role_node_residual(role, candidate)
            cost = cfg.centroid_weight * residual["centroid"] + cfg.area_weight * residual["area"] + cfg.bbox_weight * residual["bbox"]
            if "type" in cfg.matching_mode:
                cost += cfg.type_weight * residual["type"]
            if "dino" in cfg.matching_mode:
                cost += cfg.dino_weight * residual["dino"]
            if "local" in cfg.matching_mode:
                local_bonus = float((patch_summary or {}).get("boundary_patch_match_ratio", 0.0))
                cost += cfg.local_weight * max(0.0, 1.0 - local_bonus)
            matrix[role_index, candidate_index] = cost
    return matrix


def _role_node_residual(role: RolePrior, candidate: dict[str, Any]) -> dict[str, float]:
    centroid = float(np.linalg.norm(np.asarray(role.centroid_mean, dtype=np.float64) - np.asarray(_xy(candidate.get("centroid", [0.0, 0.0])), dtype=np.float64)))
    area_std = max(role.area_ratio_std, 0.01)
    area = min(2.0, abs(float(candidate.get("area_ratio", 0.0)) - role.area_ratio_mean) / area_std) / 2.0
    bbox = float(np.mean(np.abs(np.asarray(_bbox_norm(candidate.get("bbox", [0, 0, 0, 0])), dtype=np.float64) - np.asarray(role.bbox_mean, dtype=np.float64))))
    type_residual = 0.0 if str(candidate.get("normalization_type", "")) == role.dominant_normalization_type else 1.0
    role_score = _prototype_score(candidate, role.role_id)
    dino = max(0.0, 1.0 - ((role_score + 1.0) / 2.0))
    local = max(0.0, 1.0 - float(candidate.get("inside_feature_coherence", 0.0)))
    return {
        "centroid": centroid,
        "area": float(area),
        "bbox": bbox,
        "type": type_residual,
        "dino": float(dino),
        "local": float(local),
    }


def _matched_edge_residuals(
    prior: RoleGraphPrior,
    candidates: list[dict[str, Any]],
    matches: list[tuple[int, int, float]],
) -> list[dict[str, float]]:
    residuals = []
    for (left_role, left_candidate, _), (right_role, right_candidate, _) in combinations(matches, 2):
        role_key = _edge_key(prior.roles[left_role].role_id, prior.roles[right_role].role_id)
        ref_edge = prior.edge_features.get(role_key)
        if ref_edge is None:
            continue
        query_edge = _edge_feature_from_candidate(candidates[left_candidate], candidates[right_candidate])
        distance = abs(ref_edge["distance"] - query_edge["distance"])
        angle = _angle_diff(ref_edge["angle"], query_edge["angle"])
        area_ratio = abs(log(max(ref_edge["area_ratio"], 1e-8)) - log(max(query_edge["area_ratio"], 1e-8)))
        residuals.append(
            {
                "distance": float(distance),
                "angle": float(angle),
                "area_ratio": float(area_ratio),
                "residual": float(distance + angle / pi + 0.25 * area_ratio),
            }
        )
    return residuals


def _role_edge_features(roles: list[RolePrior]) -> dict[str, dict[str, float]]:
    edges = {}
    for left, right in combinations(roles, 2):
        edges[_edge_key(left.role_id, right.role_id)] = _edge_feature_from_points(
            left.centroid_mean,
            right.centroid_mean,
            left.area_ratio_mean,
            right.area_ratio_mean,
        )
    return edges


def _edge_feature_from_candidate(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return _edge_feature_from_points(
        _xy(left.get("centroid", [0.0, 0.0])),
        _xy(right.get("centroid", [0.0, 0.0])),
        float(left.get("area_ratio", 0.0)),
        float(right.get("area_ratio", 0.0)),
    )


def _edge_feature_from_points(left_xy: list[float], right_xy: list[float], left_area: float, right_area: float) -> dict[str, float]:
    dx = float(right_xy[0] - left_xy[0])
    dy = float(right_xy[1] - left_xy[1])
    return {
        "dx": dx,
        "dy": dy,
        "distance": float(sqrt(dx * dx + dy * dy)),
        "angle": float(atan2(dy, dx)),
        "area_ratio": float(max(right_area, 1e-8) / max(left_area, 1e-8)),
    }


def _patch_summary_for_mode(
    patch_by_key: dict[tuple[str, str, str, str, str, float], dict[str, Any]],
    condition: str,
    base_image_id: str,
    matching_mode: str,
) -> dict[str, Any] | None:
    if "local" not in matching_mode:
        return None
    preferred = (
        condition,
        base_image_id,
        "mask_smoothed",
        "normalized_topk_mask",
        "boundary_weighted_local",
        0.3,
    )
    fallback = (
        condition,
        base_image_id,
        "dino_only",
        "no_mask",
        "none",
        0.0,
    )
    return patch_by_key.get(preferred) or patch_by_key.get(fallback)


def _prototype_score(candidate: dict[str, Any], role_id: str) -> float:
    scores = candidate.get("prototype_scores", {})
    if isinstance(scores, dict) and role_id in scores:
        return float(scores[role_id])
    if str(candidate.get("best_prototype_id")) == role_id:
        return float(candidate.get("max_prototype_similarity", 0.0))
    return -1.0


def _large_stuff_retention(roles: list[RolePrior], matched_role_indices: set[int]) -> float:
    large_stuff_roles = [
        index
        for index, role in enumerate(roles)
        if role.area_ratio_mean >= 0.05 or "stuff" in role.dominant_normalization_type
    ]
    if not large_stuff_roles:
        return 0.0
    return float(sum(index in matched_role_indices for index in large_stuff_roles) / len(large_stuff_roles))


def _stuff_match_rate(roles: list[RolePrior], matched_role_indices: set[int]) -> float:
    stuff_roles = [
        index
        for index, role in enumerate(roles)
        if "stuff" in role.dominant_normalization_type
    ]
    if not stuff_roles:
        return 0.0
    return float(sum(index in matched_role_indices for index in stuff_roles) / len(stuff_roles))


def _fragment_burden(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    return float(
        sum(
            str(row.get("node_type")) in {"small_detail", "low_reliability_candidate"}
            or float(row.get("area_ratio", 0.0)) < 0.003
            for row in candidates
        )
        / len(candidates)
    )


def _linear_sum_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: WPS433

        rows, cols = linear_sum_assignment(cost_matrix)
        return list(zip([int(row) for row in rows], [int(col) for col in cols]))
    except Exception:  # noqa: BLE001
        return _greedy_assignment(cost_matrix)


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


def _xy(value: Any) -> list[float]:
    if isinstance(value, str):
        value = value.strip("[]").split(",")
    values = [float(item) for item in list(value)[:2]]
    return [values[0], values[1]]


def _bbox_norm(value: Any) -> list[float]:
    if isinstance(value, str):
        value = value.strip("[]").split(",")
    values = [float(item) for item in list(value)[:4]]
    x1, y1, x2, y2 = values
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    scale = max(width + height, 1.0)
    # Candidate centroids are already normalized by image size. Bboxes in the
    # descriptor are pixel coordinates, so use only scale-free shape cues here.
    return [width / scale, height / scale, width / max(height, 1.0), height / max(width, 1.0)]


def _edge_key(left: str, right: str) -> str:
    return "::".join(sorted([left, right]))


def _angle_diff(left: float, right: float) -> float:
    return float(abs((left - right + pi) % (2.0 * pi) - pi))


def _mode(values: Any) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
