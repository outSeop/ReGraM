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

    soft_assign_quantile: float = 0.20
    ot_reg: float = 0.05
    ot_marginal_penalty: float = 1.0
    min_matched_mass_for_instance: float = 0.5
    skip_signal_4_if_no_extent: bool = True
    empty_extent_penalty: float = 10.0
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
    prototype_ids = list(memory.prototypes)
    centroids = np.stack([memory.prototypes[pid].centroid for pid in prototype_ids], axis=0)
    similarities = _cosine_matrix(query_features, centroids)

    per_prototype: dict[str, PrototypeMatchingResult] = {}
    for proto_index, prototype_id in enumerate(prototype_ids):
        prototype = memory.prototypes[prototype_id]
        threshold = _prototype_threshold(prototype.within_sim_quantiles, config.soft_assign_quantile)
        active_mask = similarities[:, proto_index] > threshold
        if not np.any(active_mask):
            per_prototype[prototype_id] = PrototypeMatchingResult(
                prototype_id=prototype_id,
                query_mass=0.0,
                memory_mass=1.0,
                matching_cost=0.0,
                unmatched_query_mass=0.0,
                unmatched_memory_mass=1.0,
                matched_instance_count=0,
                expected_instance_count=float(prototype.expected_instance_count),
                instance_count_diff=float(abs(prototype.expected_instance_count)),
                matched_extent=None,
                transport_plan=np.zeros((0, _prototype_patch_count(prototype)), dtype=np.float32),
                debug={
                    "active_patch_count": 0,
                    "threshold": float(threshold),
                    "reason": "no_active_query_patches",
                },
            )
            continue

        q_features = query_features[active_mask]
        q_positions = query_positions[active_mask]
        q_similarity = np.maximum(similarities[active_mask, proto_index], 0.0).astype(np.float64)
        if float(q_similarity.sum()) <= config.eps:
            q_similarity = np.ones_like(q_similarity, dtype=np.float64)
        q_weights = q_similarity / max(float(q_similarity.sum()), config.eps)
        m_features, m_instance_ids, m_weights, expected_instance_mass = _memory_patch_bag(prototype)
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
        if float(marginal_q.sum()) > config.eps:
            matched_extent = (q_positions * marginal_q[:, None]).sum(axis=0) / max(float(marginal_q.sum()), config.eps)
        else:
            matched_extent = None
        per_prototype[prototype_id] = PrototypeMatchingResult(
            prototype_id=prototype_id,
            query_mass=1.0,
            memory_mass=1.0,
            matching_cost=matching_cost,
            unmatched_query_mass=max(0.0, 1.0 - plan_mass),
            unmatched_memory_mass=max(0.0, 1.0 - plan_mass),
            matched_instance_count=matched_instance_count,
            expected_instance_count=float(prototype.expected_instance_count),
            instance_count_diff=float(abs(matched_instance_count - prototype.expected_instance_count)),
            matched_extent=matched_extent,
            transport_plan=plan.astype(np.float32),
            debug={
                "active_patch_count": int(active_mask.sum()),
                "threshold": float(threshold),
                "plan_mass": plan_mass,
                "per_instance_mass_ratio": per_instance_ratio,
            },
        )

    spatial_violations = _spatial_violations(memory, per_prototype, config=config)
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


def _memory_patch_bag(prototype: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    features = []
    instance_ids = []
    weights = []
    expected_instance_mass = {}
    num_instances = max(len(prototype.instances), 1)
    for instance in prototype.instances:
        patch_count = max(len(instance.patch_features), 1)
        features.append(instance.patch_features)
        instance_ids.extend([instance.instance_id] * patch_count)
        per_patch_weight = 1.0 / (num_instances * patch_count)
        weights.extend([per_patch_weight] * patch_count)
        expected_instance_mass[instance.instance_id] = 1.0 / num_instances
    feature_array = np.concatenate(features, axis=0).astype(np.float32)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array = weight_array / max(float(weight_array.sum()), 1e-8)
    return feature_array, np.asarray(instance_ids), weight_array, expected_instance_mass


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
) -> dict[tuple[str, str], float]:
    violations = {}
    for pair, edge in memory.spatial_graph.items():
        left = per_prototype.get(pair[0])
        right = per_prototype.get(pair[1])
        left_extent = None if left is None else left.matched_extent
        right_extent = None if right is None else right.matched_extent
        if left_extent is None or right_extent is None:
            if not config.skip_signal_4_if_no_extent:
                violations[pair] = float(config.empty_extent_penalty)
            continue
        observed = right_extent - left_extent
        diff = observed - edge.delta_pos_mean
        cov = edge.delta_pos_cov + np.eye(2, dtype=np.float32) * 1e-6
        cov_inv = np.linalg.pinv(cov)
        violations[pair] = float(sqrt(max(float(diff.T @ cov_inv @ diff), 0.0)))
    return violations


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


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-6)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
    return left_norm @ right_norm.T
