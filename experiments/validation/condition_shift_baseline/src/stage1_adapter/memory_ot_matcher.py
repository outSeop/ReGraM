"""Memory-anchored unbalanced OT matching for W1 clean self-probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np

from stage1_adapter.memory_bank import MemoryBank


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for memory-anchored OT matching."""

    assignment_mode: str = "top_k"
    top_k_prototypes: int = 3
    min_assignment_similarity: float | None = 0.5
    soft_assign_quantile: float = 0.20
    ot_reg: float = 0.05
    ot_marginal_penalty: float = 1.0
    mass_normalization_mode: str = "balanced"
    query_count_mode: str = "memory_matched"
    query_count_similarity_threshold: float | None = None
    query_count_min_component_patches: int = 2
    min_matched_mass_for_instance: float = 0.5
    skip_signal_4_if_no_extent: bool = True
    empty_extent_penalty: float = 10.0
    spatial_signal_mode: str = "clipped_l2"
    spatial_min_occurrence_rate: float = 0.8
    spatial_min_expected_count: float = 0.7
    spatial_max_expected_count: float = 1.3
    spatial_min_edge_pairs: int = 5
    spatial_edge_tolerance: float = 0.05
    spatial_edge_cap: float = 3.0
    eps: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrototypeMatchingResult:
    """OT matching decomposition for one memory prototype."""

    prototype_id: str
    query_mass: float
    memory_mass: float
    matching_cost: float
    unmatched_query_mass: float
    unmatched_memory_mass: float
    matched_instance_count: int
    matched_instance_count_per_image: float
    expected_instance_count: float
    instance_count_diff: float
    matched_extent: np.ndarray | None
    transport_plan: np.ndarray
    debug: dict[str, Any]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe result summary without dense transport plan."""
        return {
            "prototype_id": self.prototype_id,
            "query_mass": float(self.query_mass),
            "memory_mass": float(self.memory_mass),
            "matching_cost": float(self.matching_cost),
            "unmatched_query_mass": float(self.unmatched_query_mass),
            "unmatched_memory_mass": float(self.unmatched_memory_mass),
            "matched_instance_count": int(self.matched_instance_count),
            "matched_instance_count_per_image": float(self.matched_instance_count_per_image),
            "expected_instance_count": float(self.expected_instance_count),
            "instance_count_diff": float(self.instance_count_diff),
            "matched_extent": None if self.matched_extent is None else self.matched_extent.astype(float).tolist(),
            "transport_shape": list(self.transport_plan.shape),
            "debug": self.debug,
        }


@dataclass
class AnomalyDecomposition:
    """Four-signal decomposition from memory-anchored OT matching."""

    per_prototype: dict[str, PrototypeMatchingResult]
    spatial_violations: dict[tuple[str, str], float]
    signal_1_total: float
    signal_2_total: float
    signal_3_total: float
    signal_4_total: float
    total: float
    debug: dict[str, Any]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe decomposition summary."""
        return {
            "per_prototype": {
                prototype_id: result.to_metadata_dict()
                for prototype_id, result in self.per_prototype.items()
            },
            "spatial_violations": {
                f"{left}::{right}": float(value)
                for (left, right), value in self.spatial_violations.items()
            },
            "signal_1_total": float(self.signal_1_total),
            "signal_2_total": float(self.signal_2_total),
            "signal_3_total": float(self.signal_3_total),
            "signal_4_total": float(self.signal_4_total),
            "total": float(self.total),
            "debug": self.debug,
        }


def match_query_against_memory(
    query_patch_features: np.ndarray,
    query_patch_positions: np.ndarray,
    memory: MemoryBank,
    config: MatchingConfig,
) -> AnomalyDecomposition:
    """Match raw query DINO patches to clean memory with prototype-level OT."""
    if len(query_patch_features) != len(query_patch_positions):
        raise ValueError("query_patch_features and query_patch_positions must be aligned")
    if not memory.prototypes:
        raise ValueError("memory must contain at least one prototype")
    query_features = np.asarray(query_patch_features, dtype=np.float32)
    query_positions = np.asarray(query_patch_positions, dtype=np.float32)
    total_query_patches = max(int(len(query_features)), 1)
    prototype_ids = list(memory.prototypes)
    centroids = np.stack([memory.prototypes[pid].centroid for pid in prototype_ids], axis=0)
    similarities = _cosine_matrix(query_features, centroids)
    active_by_prototype, assignment_debug = _active_assignment_mask(
        similarities,
        prototype_ids=prototype_ids,
        memory=memory,
        config=config,
    )

    per_prototype: dict[str, PrototypeMatchingResult] = {}
    for proto_index, prototype_id in enumerate(prototype_ids):
        prototype = memory.prototypes[prototype_id]
        threshold = assignment_debug["thresholds"].get(prototype_id)
        active_mask = active_by_prototype[:, proto_index]
        num_occurring_images = _num_occurring_images(prototype)
        if not np.any(active_mask):
            memory_mass = _expected_memory_mass(
                prototype,
                total_query_patches=total_query_patches,
                config=config,
            )
            instance_count_diff = (
                abs(prototype.expected_instance_count)
                if config.query_count_mode == "query_connected_components"
                else abs(prototype.expected_instance_count)
            )
            per_prototype[prototype_id] = PrototypeMatchingResult(
                prototype_id=prototype_id,
                query_mass=0.0,
                memory_mass=memory_mass,
                matching_cost=0.0,
                unmatched_query_mass=0.0,
                unmatched_memory_mass=memory_mass,
                matched_instance_count=0,
                matched_instance_count_per_image=0.0,
                expected_instance_count=float(prototype.expected_instance_count),
                instance_count_diff=float(instance_count_diff),
                matched_extent=None,
                transport_plan=np.zeros((0, _prototype_patch_count(prototype)), dtype=np.float32),
                debug={
                    "active_patch_count": 0,
                    "assignment_mode": config.assignment_mode,
                    "top_k_prototypes": int(config.top_k_prototypes),
                    "threshold": None if threshold is None else float(threshold),
                    "num_occurring_images": num_occurring_images,
                    "mass_normalization_mode": config.mass_normalization_mode,
                    "query_count_mode": config.query_count_mode,
                    "query_instance_count": 0,
                    "reason": "no_active_query_patches",
                },
            )
            continue

        q_features = query_features[active_mask]
        q_positions = query_positions[active_mask]
        q_similarity = np.maximum(similarities[active_mask, proto_index], 0.0).astype(np.float64)
        q_weights = _query_patch_weights(
            q_similarity,
            total_query_patches=total_query_patches,
            config=config,
        )
        m_features, m_instance_ids, m_weights, expected_instance_mass = _memory_patch_bag(
            prototype,
            total_query_patches=total_query_patches,
            config=config,
        )
        query_mass = float(q_weights.sum())
        memory_mass = float(m_weights.sum())
        cost = 1.0 - _cosine_matrix(q_features, m_features)
        plan = _sinkhorn_unbalanced(q_weights, m_weights, cost, config=config)
        marginal_q = plan.sum(axis=1)
        marginal_m = plan.sum(axis=0)
        plan_mass = float(plan.sum())
        matching_cost = float(np.sum(plan * cost))
        per_instance_ratio = _per_instance_mass_ratio(
            marginal_m,
            m_instance_ids,
            expected_instance_mass=expected_instance_mass,
        )
        matched_instance_count = int(
            sum(value > config.min_matched_mass_for_instance for value in per_instance_ratio.values())
        )
        if config.query_count_mode == "query_connected_components":
            query_instance_count = _query_instance_count(
                active_mask,
                similarities[:, proto_index],
                query_positions,
                config=config,
            )
            matched_instance_count = int(query_instance_count)
            matched_instance_count_per_image = float(query_instance_count)
            instance_count_diff = float(abs(query_instance_count - prototype.expected_instance_count))
        elif config.query_count_mode == "memory_matched":
            query_instance_count = None
            matched_instance_count_per_image = matched_instance_count / max(float(num_occurring_images), 1.0)
            instance_count_diff = float(abs(matched_instance_count_per_image - prototype.expected_instance_count))
        else:
            raise ValueError(f"unsupported query_count_mode: {config.query_count_mode}")
        if float(marginal_q.sum()) > config.eps:
            matched_extent = (q_positions * marginal_q[:, None]).sum(axis=0) / max(float(marginal_q.sum()), config.eps)
        else:
            matched_extent = None
        per_prototype[prototype_id] = PrototypeMatchingResult(
            prototype_id=prototype_id,
            query_mass=query_mass,
            memory_mass=memory_mass,
            matching_cost=matching_cost,
            unmatched_query_mass=max(0.0, query_mass - plan_mass),
            unmatched_memory_mass=max(0.0, memory_mass - plan_mass),
            matched_instance_count=matched_instance_count,
            matched_instance_count_per_image=float(matched_instance_count_per_image),
            expected_instance_count=float(prototype.expected_instance_count),
            instance_count_diff=instance_count_diff,
            matched_extent=matched_extent,
            transport_plan=plan.astype(np.float32),
            debug={
                "active_patch_count": int(active_mask.sum()),
                "assignment_mode": config.assignment_mode,
                "top_k_prototypes": int(config.top_k_prototypes),
                "threshold": None if threshold is None else float(threshold),
                "num_occurring_images": num_occurring_images,
                "mass_normalization_mode": config.mass_normalization_mode,
                "query_count_mode": config.query_count_mode,
                "query_instance_count": None if query_instance_count is None else int(query_instance_count),
                "plan_mass": plan_mass,
                "matched_extent_weighting": "ot_query_marginal",
                "per_instance_mass_ratio": per_instance_ratio,
            },
        )

    spatial_violations, spatial_debug = _spatial_violations(memory, per_prototype, config=config)
    signal_1 = float(sum(result.unmatched_query_mass for result in per_prototype.values()))
    signal_2 = float(sum(result.unmatched_memory_mass for result in per_prototype.values()))
    signal_3 = float(sum(result.instance_count_diff for result in per_prototype.values()))
    signal_4 = float(sum(spatial_violations.values()))
    total = signal_1 + signal_2 + signal_3 + signal_4
    return AnomalyDecomposition(
        per_prototype=per_prototype,
        spatial_violations=spatial_violations,
        signal_1_total=signal_1,
        signal_2_total=signal_2,
        signal_3_total=signal_3,
        signal_4_total=signal_4,
        total=float(total),
        debug={
            "num_query_patches": int(len(query_features)),
            "num_prototypes": len(memory.prototypes),
            "assignment": assignment_debug,
            "spatial": spatial_debug,
            "matching_config": config.to_dict(),
        },
    )


def query_patch_grid_from_feature_map(feature_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten HxWxD features and return normalized HxW patch positions."""
    height, width, dim = feature_map.shape
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    positions = np.stack(
        [
            xs.astype(np.float32) / max(width - 1, 1),
            ys.astype(np.float32) / max(height - 1, 1),
        ],
        axis=-1,
    )
    return feature_map.reshape(height * width, dim).astype(np.float32), positions.reshape(height * width, 2)


def decomposition_summary(result: AnomalyDecomposition) -> dict[str, Any]:
    """Return compact W1 display summary."""
    return {
        "signal_1_total": result.signal_1_total,
        "signal_2_total": result.signal_2_total,
        "signal_3_total": result.signal_3_total,
        "signal_4_total": result.signal_4_total,
        "total": result.total,
        "num_spatial_violations": len(result.spatial_violations),
        "num_prototypes": len(result.per_prototype),
    }


def active_assignment_mask(
    query_patch_features: np.ndarray,
    prototype_centroids: np.ndarray,
    prototype_ids: list[str],
    memory: MemoryBank,
    config: MatchingConfig,
) -> np.ndarray:
    """Return query-patch/prototype active assignment mask for visualization."""
    similarities = _cosine_matrix(query_patch_features, prototype_centroids)
    active, _ = _active_assignment_mask(
        similarities,
        prototype_ids=prototype_ids,
        memory=memory,
        config=config,
    )
    return active


def _sinkhorn_unbalanced(a: np.ndarray, b: np.ndarray, cost: np.ndarray, *, config: MatchingConfig) -> np.ndarray:
    try:
        import ot  # type: ignore  # noqa: WPS433

        return np.asarray(
            ot.unbalanced.sinkhorn_unbalanced(
                a=a.astype(np.float64),
                b=b.astype(np.float64),
                M=cost.astype(np.float64),
                reg=float(config.ot_reg),
                reg_m=float(config.ot_marginal_penalty),
            ),
            dtype=np.float64,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "POT is required for W1 memory OT matching. Install with `pip install pot`."
        ) from exc


def _active_assignment_mask(
    similarities: np.ndarray,
    *,
    prototype_ids: list[str],
    memory: MemoryBank,
    config: MatchingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.assignment_mode == "threshold":
        active = np.zeros_like(similarities, dtype=bool)
        thresholds = {}
        for proto_index, prototype_id in enumerate(prototype_ids):
            threshold = _prototype_threshold(
                memory.prototypes[prototype_id].within_sim_quantiles,
                config.soft_assign_quantile,
            )
            thresholds[prototype_id] = float(threshold)
            active[:, proto_index] = similarities[:, proto_index] > threshold
        return active, {
            "assignment_mode": config.assignment_mode,
            "thresholds": thresholds,
            "top_k_prototypes": None,
        }
    if config.assignment_mode == "top_k":
        k = max(1, min(int(config.top_k_prototypes), similarities.shape[1]))
        active = np.zeros_like(similarities, dtype=bool)
        top_indices = np.argsort(-similarities, axis=1)[:, :k]
        row_indices = np.arange(similarities.shape[0])[:, None]
        active[row_indices, top_indices] = True
        if config.min_assignment_similarity is not None:
            active &= similarities >= float(config.min_assignment_similarity)
        return active, {
            "assignment_mode": config.assignment_mode,
            "thresholds": {prototype_id: None for prototype_id in prototype_ids},
            "top_k_prototypes": k,
            "min_assignment_similarity": config.min_assignment_similarity,
            "active_assignments": int(active.sum()),
        }
    raise ValueError(f"unsupported assignment_mode: {config.assignment_mode}")


def _query_patch_weights(
    q_similarity: np.ndarray,
    *,
    total_query_patches: int,
    config: MatchingConfig,
) -> np.ndarray:
    """Build query-side OT mass without always forcing unit prototype mass."""
    if config.mass_normalization_mode == "balanced":
        if float(q_similarity.sum()) <= config.eps:
            q_similarity = np.ones_like(q_similarity, dtype=np.float64)
        return q_similarity / max(float(q_similarity.sum()), config.eps)
    if config.mass_normalization_mode == "area":
        weights = q_similarity.astype(np.float64) / max(float(total_query_patches), 1.0)
        if float(weights.sum()) <= config.eps:
            weights = np.ones_like(q_similarity, dtype=np.float64) / max(float(total_query_patches), 1.0)
        return weights
    raise ValueError(f"unsupported mass_normalization_mode: {config.mass_normalization_mode}")


def _memory_patch_bag(
    prototype: Any,
    *,
    total_query_patches: int,
    config: MatchingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    features = []
    instance_ids = []
    weights = []
    expected_instance_mass = {}
    num_instances = max(len(prototype.instances), 1)
    num_occurring_images = max(_num_occurring_images(prototype), 1)
    for instance in prototype.instances:
        patch_count = max(len(instance.patch_features), 1)
        features.append(instance.patch_features)
        instance_ids.extend([instance.instance_id] * patch_count)
        if config.mass_normalization_mode == "balanced":
            instance_mass = 1.0 / num_instances
        elif config.mass_normalization_mode == "area":
            instance_mass = patch_count / max(float(total_query_patches * num_occurring_images), 1.0)
        else:
            raise ValueError(f"unsupported mass_normalization_mode: {config.mass_normalization_mode}")
        per_patch_weight = instance_mass / patch_count
        weights.extend([per_patch_weight] * patch_count)
        expected_instance_mass[instance.instance_id] = float(instance_mass)
    feature_array = np.concatenate(features, axis=0).astype(np.float32)
    weight_array = np.asarray(weights, dtype=np.float64)
    if config.mass_normalization_mode == "balanced":
        weight_array = weight_array / max(float(weight_array.sum()), config.eps)
    return feature_array, np.asarray(instance_ids), weight_array, expected_instance_mass


def _expected_memory_mass(prototype: Any, *, total_query_patches: int, config: MatchingConfig) -> float:
    if config.mass_normalization_mode == "balanced":
        return 1.0
    if config.mass_normalization_mode == "area":
        return float(
            sum(len(instance.patch_features) for instance in prototype.instances)
            / max(float(total_query_patches * max(_num_occurring_images(prototype), 1)), 1.0)
        )
    raise ValueError(f"unsupported mass_normalization_mode: {config.mass_normalization_mode}")


def _query_instance_count(
    active_mask: np.ndarray,
    prototype_similarity: np.ndarray,
    query_positions: np.ndarray,
    *,
    config: MatchingConfig,
) -> int:
    threshold = (
        float(config.query_count_similarity_threshold)
        if config.query_count_similarity_threshold is not None
        else config.min_assignment_similarity
    )
    if threshold is None:
        threshold = -1.0
    response_mask = active_mask & (prototype_similarity >= float(threshold))
    if not np.any(response_mask):
        return 0
    response_grid = _patch_mask_to_grid(response_mask, query_positions)
    return _connected_component_count(
        response_grid,
        min_component_patches=max(1, int(config.query_count_min_component_patches)),
    )


def _patch_mask_to_grid(mask: np.ndarray, positions: np.ndarray) -> np.ndarray:
    xs = np.unique(np.round(positions[:, 0], 6))
    ys = np.unique(np.round(positions[:, 1], 6))
    width = max(int(len(xs)), 1)
    height = max(int(len(ys)), 1)
    grid = np.zeros((height, width), dtype=bool)
    x_index = np.rint(np.clip(positions[:, 0], 0.0, 1.0) * max(width - 1, 0)).astype(int)
    y_index = np.rint(np.clip(positions[:, 1], 0.0, 1.0) * max(height - 1, 0)).astype(int)
    grid[y_index[mask], x_index[mask]] = True
    return grid


def _connected_component_count(mask: np.ndarray, *, min_component_patches: int) -> int:
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = cy + dy
                        nx = cx + dx
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if size >= min_component_patches:
                count += 1
    return count


def _per_instance_mass_ratio(
    marginal_m: np.ndarray,
    instance_ids: np.ndarray,
    *,
    expected_instance_mass: dict[str, float],
) -> dict[str, float]:
    mass = {instance_id: 0.0 for instance_id in expected_instance_mass}
    for value, instance_id in zip(marginal_m, instance_ids, strict=True):
        mass[str(instance_id)] = mass.get(str(instance_id), 0.0) + float(value)
    return {
        instance_id: float(value / max(expected_instance_mass.get(instance_id, 1.0), 1e-8))
        for instance_id, value in mass.items()
    }


def _spatial_violations(
    memory: MemoryBank,
    per_prototype: dict[str, PrototypeMatchingResult],
    *,
    config: MatchingConfig,
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    violations = {}
    skipped_edges: list[dict[str, Any]] = []
    skipped_by_reason: dict[str, int] = {}
    eligible_edges = 0
    for pair, edge in memory.spatial_graph.items():
        eligible, reason = _edge_is_required_role(memory, pair, edge, config=config)
        if not eligible:
            skipped_edges.append(
                {
                    "prototype_i": pair[0],
                    "prototype_j": pair[1],
                    "reason": reason,
                    "num_edge_pairs": int(edge.num_pairs),
                }
            )
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
            continue
        eligible_edges += 1
        left = per_prototype.get(pair[0])
        right = per_prototype.get(pair[1])
        left_extent = None if left is None else left.matched_extent
        right_extent = None if right is None else right.matched_extent
        if left_extent is None or right_extent is None:
            if not config.skip_signal_4_if_no_extent:
                violations[pair] = float(config.empty_extent_penalty)
            skipped_edges.append(
                {
                    "prototype_i": pair[0],
                    "prototype_j": pair[1],
                    "reason": "missing_matched_extent",
                    "num_edge_pairs": int(edge.num_pairs),
                }
            )
            skipped_by_reason["missing_matched_extent"] = skipped_by_reason.get("missing_matched_extent", 0) + 1
            continue
        observed = right_extent - left_extent
        diff = observed - edge.delta_pos_mean
        if config.spatial_signal_mode == "mahalanobis":
            cov = edge.delta_pos_cov + np.eye(2, dtype=np.float32) * 1e-6
            cov_inv = np.linalg.pinv(cov)
            violations[pair] = float(sqrt(max(float(diff.T @ cov_inv @ diff), 0.0)))
        elif config.spatial_signal_mode == "clipped_l2":
            memory_edge_length = float(np.linalg.norm(edge.delta_pos_mean))
            raw_error = float(np.linalg.norm(diff))
            normalized_error = raw_error / max(
                float(config.spatial_edge_tolerance) + memory_edge_length,
                float(config.eps),
            )
            violations[pair] = float(min(normalized_error, float(config.spatial_edge_cap)))
        else:
            raise ValueError(f"unsupported spatial_signal_mode: {config.spatial_signal_mode}")
    debug = {
        "spatial_signal_mode": config.spatial_signal_mode,
        "num_memory_edges": len(memory.spatial_graph),
        "num_required_role_edges": eligible_edges,
        "num_scored_edges": len(violations),
        "num_skipped_edges": len(skipped_edges),
        "skipped_by_reason": skipped_by_reason,
        "skipped_edges": skipped_edges,
        "required_role_filter": {
            "min_occurrence_rate": float(config.spatial_min_occurrence_rate),
            "min_expected_count": float(config.spatial_min_expected_count),
            "max_expected_count": float(config.spatial_max_expected_count),
            "min_edge_pairs": int(config.spatial_min_edge_pairs),
        },
    }
    return violations, debug


def _edge_is_required_role(
    memory: MemoryBank,
    pair: tuple[str, str],
    edge: Any,
    *,
    config: MatchingConfig,
) -> tuple[bool, str]:
    if int(edge.num_pairs) < int(config.spatial_min_edge_pairs):
        return False, "too_few_edge_pairs"
    left = memory.prototypes[pair[0]]
    right = memory.prototypes[pair[1]]
    for prototype in (left, right):
        if float(prototype.occurrence_rate) < float(config.spatial_min_occurrence_rate):
            return False, "low_occurrence_role"
        expected_count = float(prototype.expected_instance_count)
        if expected_count < float(config.spatial_min_expected_count):
            return False, "low_expected_count_role"
        if expected_count > float(config.spatial_max_expected_count):
            return False, "multi_instance_role"
    return True, "required_role_edge"


def _prototype_threshold(quantiles: dict[float, float], quantile: float) -> float:
    if quantile in quantiles:
        return float(quantiles[quantile])
    if str(quantile) in quantiles:
        return float(quantiles[str(quantile)])  # type: ignore[index]
    if not quantiles:
        return -1.0
    closest = min(quantiles, key=lambda key: abs(float(key) - float(quantile)))
    return float(quantiles[closest])


def _prototype_patch_count(prototype: Any) -> int:
    return int(sum(len(instance.patch_features) for instance in prototype.instances))


def _num_occurring_images(prototype: Any) -> int:
    return int(len({instance.source_image_id for instance in prototype.instances}))


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-6)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
    return left_norm @ right_norm.T
