"""Prototype memory and reliability scoring for Stage 1 component proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from typing import Any

import numpy as np

from stage1_adapter.descriptors import CandidateDescriptor, descriptor_vector


@dataclass(frozen=True)
class ComponentPrototype:
    prototype_id: str
    member_count: int
    mean_vector: list[float]
    inside_mean: list[float]
    ring_mean: list[float]
    centroid_mean: list[float]
    area_ratio_mean: float
    area_ratio_std: float
    coherence_mean: float
    occurrence_rate: float
    debug: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_component_prototypes(
    descriptors: list[CandidateDescriptor],
    *,
    max_prototypes: int = 8,
    min_members: int = 1,
) -> list[ComponentPrototype]:
    """Build deterministic memory prototypes from clean normal descriptors."""
    if not descriptors:
        return []
    vectors = _standardize(np.stack([descriptor_vector(item) for item in descriptors], axis=0))
    k = max(1, min(int(max_prototypes), len(descriptors)))
    labels = _deterministic_kmeans(vectors, k=k)
    image_count = max(1, len({item.source_image_id for item in descriptors}))
    prototypes: list[ComponentPrototype] = []
    for cluster_index in range(k):
        member_indices = np.where(labels == cluster_index)[0]
        if len(member_indices) < min_members:
            continue
        members = [descriptors[int(index)] for index in member_indices]
        raw_vectors = np.stack([descriptor_vector(item) for item in members], axis=0)
        inside_means = np.stack([np.asarray(item.inside_mean, dtype=np.float32) for item in members], axis=0)
        ring_means = np.stack([np.asarray(item.ring_mean, dtype=np.float32) for item in members], axis=0)
        area_ratios = np.asarray([item.area_ratio for item in members], dtype=np.float32)
        centroids = np.asarray([item.centroid for item in members], dtype=np.float32)
        prototypes.append(
            ComponentPrototype(
                prototype_id=f"prototype_{len(prototypes):03d}",
                member_count=len(members),
                mean_vector=raw_vectors.mean(axis=0).astype(float).tolist(),
                inside_mean=inside_means.mean(axis=0).astype(float).tolist(),
                ring_mean=ring_means.mean(axis=0).astype(float).tolist(),
                centroid_mean=centroids.mean(axis=0).astype(float).tolist(),
                area_ratio_mean=float(area_ratios.mean()),
                area_ratio_std=float(area_ratios.std()),
                coherence_mean=float(np.mean([item.inside_feature_coherence for item in members])),
                occurrence_rate=float(len({item.source_image_id for item in members}) / image_count),
                debug={
                    "member_candidate_ids": [item.candidate_id for item in members],
                    "member_source_image_ids": sorted({item.source_image_id for item in members}),
                },
            )
        )
    prototypes.sort(key=lambda item: (-item.member_count, item.prototype_id))
    return [
        ComponentPrototype(
            prototype_id=f"prototype_{index:03d}",
            member_count=item.member_count,
            mean_vector=item.mean_vector,
            inside_mean=item.inside_mean,
            ring_mean=item.ring_mean,
            centroid_mean=item.centroid_mean,
            area_ratio_mean=item.area_ratio_mean,
            area_ratio_std=item.area_ratio_std,
            coherence_mean=item.coherence_mean,
            occurrence_rate=item.occurrence_rate,
            debug=item.debug,
        )
        for index, item in enumerate(prototypes)
    ]


def score_candidate_against_prototypes(
    descriptor: CandidateDescriptor,
    prototypes: list[ComponentPrototype],
    *,
    too_large_area_ratio: float = 0.55,
    fragment_area_ratio: float = 0.0005,
    supermask_containment_threshold: int = 3,
) -> dict[str, Any]:
    """Assign prototype compatibility, reliability, and proposal type."""
    if not prototypes:
        return _invalid_score(descriptor, reason="no_prototypes")
    vector = descriptor_vector(descriptor)
    prototype_vectors = np.stack([np.asarray(item.mean_vector, dtype=np.float32) for item in prototypes], axis=0)
    similarities = np.asarray([_cosine(vector, proto_vector) for proto_vector in prototype_vectors], dtype=np.float32)
    probabilities = _softmax(similarities)
    best_index = int(np.argmax(similarities))
    best = prototypes[best_index]
    max_similarity = float(similarities[best_index])
    sorted_similarities = np.sort(similarities)
    top2_similarity = float(sorted_similarities[-2]) if len(sorted_similarities) > 1 else max_similarity
    prototype_margin = float(max_similarity - top2_similarity)
    entropy = _entropy(probabilities)
    entropy_norm = entropy / max(log(len(prototypes)), 1e-6)
    geometry_score = _geometry_compatibility(descriptor, best)
    invalid_reasons = _invalid_reasons(
        descriptor,
        too_large_area_ratio=too_large_area_ratio,
        fragment_area_ratio=fragment_area_ratio,
        supermask_containment_threshold=supermask_containment_threshold,
    )
    similarity_score = _clamp01((max_similarity + 1.0) / 2.0)
    margin_score = _clamp01(0.5 + 2.0 * prototype_margin)
    entropy_penalty = 0.10 * _clamp01(entropy_norm)
    reliability = (
        0.45 * similarity_score
        + 0.20 * margin_score
        + 0.20 * geometry_score
        + 0.15 * _clamp01(descriptor.inside_feature_coherence)
        - entropy_penalty
    )
    reliability = _clamp01(reliability)
    if invalid_reasons:
        reliability *= 0.1
    node_type = _node_type(descriptor, invalid_reasons=invalid_reasons, reliability=reliability)
    return {
        "candidate_id": descriptor.candidate_id,
        "source_image_id": descriptor.source_image_id,
        "node_type": node_type,
        "best_prototype_id": best.prototype_id,
        "reliability": float(reliability),
        "max_prototype_similarity": max_similarity,
        "top2_prototype_similarity": top2_similarity,
        "prototype_margin": prototype_margin,
        "prototype_entropy": float(entropy),
        "prototype_entropy_norm": float(entropy_norm),
        "geometry_compatibility": float(geometry_score),
        "inside_feature_coherence": float(descriptor.inside_feature_coherence),
        "contained_neighbor_count": descriptor.contained_neighbor_count,
        "overlap_neighbor_count": descriptor.overlap_neighbor_count,
        "area_ratio": descriptor.area_ratio,
        "bbox": descriptor.bbox,
        "centroid": descriptor.centroid,
        "prototype_scores": {
            prototypes[index].prototype_id: float(similarities[index])
            for index in range(len(prototypes))
        },
        "debug": {
            "invalid_reasons": invalid_reasons,
            "proposal_source": descriptor.source,
        },
    }


def summarize_adapter_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize reliability and node type distribution."""
    node_counts: dict[str, int] = {}
    reliabilities = []
    for score in scores:
        node_type = str(score.get("node_type", "unknown"))
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
        reliabilities.append(float(score.get("reliability", 0.0)))
    return {
        "num_candidates": len(scores),
        "node_type_counts": node_counts,
        "mean_reliability": float(np.mean(reliabilities)) if reliabilities else 0.0,
        "median_reliability": float(np.median(reliabilities)) if reliabilities else 0.0,
    }


def _invalid_score(descriptor: CandidateDescriptor, *, reason: str) -> dict[str, Any]:
    return {
        "candidate_id": descriptor.candidate_id,
        "source_image_id": descriptor.source_image_id,
        "node_type": "invalid_fragment",
        "best_prototype_id": None,
        "reliability": 0.0,
        "debug": {"invalid_reasons": [reason], "proposal_source": descriptor.source},
    }


def _deterministic_kmeans(vectors: np.ndarray, *, k: int, iterations: int = 20) -> np.ndarray:
    order = np.argsort(vectors[:, 0], kind="mergesort")
    centers = vectors[order[np.linspace(0, len(order) - 1, num=k, dtype=int)]]
    labels = np.zeros(len(vectors), dtype=np.int64)
    for _ in range(iterations):
        distances = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_index in range(k):
            members = vectors[labels == cluster_index]
            if len(members):
                centers[cluster_index] = members.mean(axis=0)
    return labels


def _standardize(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(axis=0, keepdims=True)
    std = vectors.std(axis=0, keepdims=True)
    return (vectors - mean) / np.maximum(std, 1e-6)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    denom = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-6)
    return float(np.dot(left, right) / denom)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(), 1e-6)


def _entropy(probabilities: np.ndarray) -> float:
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-9))))


def _geometry_compatibility(descriptor: CandidateDescriptor, prototype: ComponentPrototype) -> float:
    centroid_dist = float(np.linalg.norm(np.asarray(descriptor.centroid) - np.asarray(prototype.centroid_mean)))
    area_std = max(float(prototype.area_ratio_std), 0.01)
    area_z = abs(float(descriptor.area_ratio) - float(prototype.area_ratio_mean)) / area_std
    return _clamp01(exp(-2.0 * centroid_dist) * exp(-0.25 * area_z))


def _invalid_reasons(
    descriptor: CandidateDescriptor,
    *,
    too_large_area_ratio: float,
    fragment_area_ratio: float,
    supermask_containment_threshold: int,
) -> list[str]:
    reasons: list[str] = []
    if descriptor.area_ratio > too_large_area_ratio:
        reasons.append("too_large_container_or_supermask")
    if descriptor.area_ratio < fragment_area_ratio:
        reasons.append("too_small_fragment")
    if descriptor.contained_neighbor_count >= supermask_containment_threshold:
        reasons.append("contains_many_neighbors")
    if descriptor.fill_ratio < 0.05:
        reasons.append("very_sparse_mask")
    return reasons


def _node_type(descriptor: CandidateDescriptor, *, invalid_reasons: list[str], reliability: float) -> str:
    if "too_large_container_or_supermask" in invalid_reasons:
        return "invalid_supermask"
    if invalid_reasons:
        return "invalid_fragment"
    if reliability < 0.2:
        return "low_reliability_candidate"
    if descriptor.area_ratio < 0.01:
        return "small_detail"
    return "valid_component"


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
