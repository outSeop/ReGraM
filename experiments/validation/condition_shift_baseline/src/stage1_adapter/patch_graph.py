"""Mask-aware patch graph utilities for Stage 1 component boundary adaptation.

This module intentionally stops at a probe-ready representation. It does not
train a GNN or anomaly detector. It builds the inputs needed for masked patch
prototype prediction and provides a deterministic 8-neighbor baseline that can
be inspected before adding a learnable adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

from stage1_adapter.prototypes import ComponentPrototype


@dataclass(frozen=True)
class PatchGraphConfig:
    """Configuration for reliable mask gating and local patch prediction."""

    reliable_selection_mode: str = "top_k"
    top_k_reliable_masks: int = 8
    prototype_top_k: int = 1
    quantile: float = 0.75
    min_reliability: float = 0.35
    mask_usage_field: str = "use_for_patch_graph"
    valid_node_types: tuple[str, ...] = ("valid_component", "small_detail")
    hard_excluded_node_types: tuple[str, ...] = ("invalid_supermask", "invalid_fragment")
    max_area_ratio: float = 0.30
    min_area_ratio: float = 0.001
    same_mask_bonus: float = 1.0
    cross_boundary_weight: float = 0.25
    unlabeled_neighbor_weight: float = 0.5
    prototype_temperature: float = 0.07
    eps: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchGraphProbeResult:
    """Serializable summary plus dense maps for notebook visualization."""

    summary: dict[str, Any]
    membership: np.ndarray
    reliable_mask: np.ndarray
    teacher_assignment: np.ndarray
    neighbor_assignment: np.ndarray
    teacher_probabilities: np.ndarray
    neighbor_probabilities: np.ndarray


def build_reliable_patch_membership(
    *,
    raw_masks: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
    feature_shape: tuple[int, int],
    image_shape: tuple[int, int],
    config: PatchGraphConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Project reliable candidate masks onto the DINO patch grid.

    The returned membership map contains an integer reliable-mask index for
    each patch, or -1 for unlabeled patches. Overlapping masks are resolved by
    candidate reliability, so invalid super-masks do not overwrite better
    component proposals.
    """
    cfg = config or PatchGraphConfig()
    score_by_id = {str(score.get("candidate_id")): score for score in candidate_scores}
    selected_ids, selection_debug = select_reliable_candidate_ids(candidate_scores, config=cfg)
    membership = np.full(feature_shape, -1, dtype=np.int32)
    reliability_map = np.zeros(feature_shape, dtype=np.float32)
    reliable_records: list[dict[str, Any]] = []
    sorted_masks = sorted(
        enumerate(raw_masks),
        key=lambda item: float(score_by_id.get(_mask_id(item[1], item[0]), {}).get("reliability", 0.0)),
    )
    for raw_index, raw_mask in sorted_masks:
        mask_id = _mask_id(raw_mask, raw_index)
        score = score_by_id.get(mask_id)
        if score is None or mask_id not in selected_ids:
            continue
        if not bool(raw_mask.get(cfg.mask_usage_field, True)):
            continue
        patch_mask = _resize_mask(np.asarray(raw_mask["mask"], dtype=bool), feature_shape)
        reliability = float(score.get("reliability", 0.0))
        record_index = len(reliable_records)
        update = np.logical_and(patch_mask, reliability >= reliability_map)
        membership[update] = record_index
        reliability_map[update] = reliability
        reliable_records.append(
            {
                "record_index": record_index,
                "mask_id": mask_id,
                "candidate_id": mask_id,
                "node_type": score.get("node_type"),
                "reliability": reliability,
                "area_ratio": float(score.get("area_ratio", 0.0)),
            }
        )
    debug = {
        "image_shape": list(image_shape),
        "feature_shape": list(feature_shape),
        "num_raw_masks": len(raw_masks),
        "num_reliable_masks": len(reliable_records),
        "reliable_records": reliable_records,
        "selection": selection_debug,
    }
    return membership, reliability_map, debug


def select_reliable_candidate_ids(
    candidate_scores: list[dict[str, Any]],
    *,
    config: PatchGraphConfig | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Select mask ids using hard invalid filtering followed by threshold/top-k."""
    cfg = config or PatchGraphConfig()
    if cfg.reliable_selection_mode == "none":
        return set(), {
            "mode": cfg.reliable_selection_mode,
            "num_input_scores": len(candidate_scores),
            "num_after_hard_filter": 0,
            "num_selected": 0,
            "selected_candidate_ids": [],
        }
    candidates = [score for score in candidate_scores if _passes_hard_filter(score, cfg)]
    if cfg.reliable_selection_mode == "threshold":
        selected = [
            score
            for score in candidates
            if str(score.get("node_type")) in set(cfg.valid_node_types)
            and float(score.get("reliability", 0.0)) >= cfg.min_reliability
        ]
    elif cfg.reliable_selection_mode == "top_k":
        selected = sorted(candidates, key=_candidate_rank_key, reverse=True)[: max(0, int(cfg.top_k_reliable_masks))]
    elif cfg.reliable_selection_mode == "prototype_top_k":
        selected = _select_prototype_top_k(candidates, top_k=max(1, int(cfg.prototype_top_k)))
    elif cfg.reliable_selection_mode == "quantile":
        selected = _select_quantile(candidates, quantile=float(cfg.quantile))
    else:
        raise ValueError(f"unsupported reliable_selection_mode: {cfg.reliable_selection_mode}")
    selected_ids = {str(score.get("candidate_id")) for score in selected}
    debug = {
        "mode": cfg.reliable_selection_mode,
        "num_input_scores": len(candidate_scores),
        "num_after_hard_filter": len(candidates),
        "num_selected": len(selected_ids),
        "selected_candidate_ids": sorted(selected_ids),
    }
    return selected_ids, debug


def assign_patch_prototypes(
    *,
    feature_map: np.ndarray,
    prototypes: list[ComponentPrototype],
    config: PatchGraphConfig | None = None,
) -> np.ndarray:
    """Assign each patch to clean-normal component prototypes with softmax."""
    cfg = config or PatchGraphConfig()
    if not prototypes:
        raise ValueError("at least one prototype is required")
    proto = np.stack([np.asarray(item.inside_mean, dtype=np.float32) for item in prototypes], axis=0)
    flat = feature_map.reshape(-1, feature_map.shape[-1]).astype(np.float32)
    logits = _cosine_matrix(flat, proto) / max(float(cfg.prototype_temperature), cfg.eps)
    probabilities = _softmax(logits, axis=1)
    return probabilities.reshape(feature_map.shape[0], feature_map.shape[1], len(prototypes))


def predict_masked_patch_prototypes_from_neighbors(
    *,
    teacher_probabilities: np.ndarray,
    membership: np.ndarray,
    reliability_map: np.ndarray,
    config: PatchGraphConfig | None = None,
) -> np.ndarray:
    """Predict masked target prototype distribution from 8-neighbor context."""
    cfg = config or PatchGraphConfig()
    height, width, num_prototypes = teacher_probabilities.shape
    prediction = np.zeros_like(teacher_probabilities, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            weighted = np.zeros(num_prototypes, dtype=np.float32)
            total_weight = 0.0
            target_membership = int(membership[y, x])
            for ny, nx in _neighbor_coords(y, x, height, width):
                neighbor_membership = int(membership[ny, nx])
                weight = _edge_weight(
                    target_membership=target_membership,
                    neighbor_membership=neighbor_membership,
                    neighbor_reliability=float(reliability_map[ny, nx]),
                    config=cfg,
                )
                weighted += weight * teacher_probabilities[ny, nx]
                total_weight += weight
            if total_weight <= cfg.eps:
                prediction[y, x] = teacher_probabilities[y, x]
            else:
                prediction[y, x] = weighted / total_weight
    return prediction


def run_masked_patch_prototype_probe(
    *,
    feature_map: np.ndarray,
    raw_masks: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
    prototypes: list[ComponentPrototype],
    image_shape: tuple[int, int],
    config: PatchGraphConfig | None = None,
) -> PatchGraphProbeResult:
    """Run a deterministic masked patch prototype prediction sanity check."""
    cfg = config or PatchGraphConfig()
    membership, reliability_map, membership_debug = build_reliable_patch_membership(
        raw_masks=raw_masks,
        candidate_scores=candidate_scores,
        feature_shape=feature_map.shape[:2],
        image_shape=image_shape,
        config=cfg,
    )
    teacher = assign_patch_prototypes(feature_map=feature_map, prototypes=prototypes, config=cfg)
    neighbor = predict_masked_patch_prototypes_from_neighbors(
        teacher_probabilities=teacher,
        membership=membership,
        reliability_map=reliability_map,
        config=cfg,
    )
    reliable_mask = membership >= 0
    region_masks = patch_region_masks(membership)
    teacher_assignment = np.argmax(teacher, axis=2).astype(np.int32)
    neighbor_assignment = np.argmax(neighbor, axis=2).astype(np.int32)
    edge_summary = summarize_patch_edges(membership)
    edge_metrics = summarize_patch_edge_metrics(
        membership=membership,
        teacher_probabilities=teacher,
        prediction_probabilities=neighbor,
    )
    kl_map = _kl_divergence(teacher, neighbor)
    if np.any(reliable_mask):
        reliable_kl = float(np.mean(kl_map[reliable_mask]))
        reliable_acc = float(np.mean(teacher_assignment[reliable_mask] == neighbor_assignment[reliable_mask]))
    else:
        reliable_kl = 0.0
        reliable_acc = 0.0
    summary = {
        "num_patches": int(feature_map.shape[0] * feature_map.shape[1]),
        "num_reliable_patches": int(reliable_mask.sum()),
        "reliable_patch_ratio": float(reliable_mask.mean()),
        "teacher_entropy_mean": float(np.mean(_entropy(teacher))),
        "neighbor_prediction_kl_mean": float(np.mean(kl_map)),
        "neighbor_prediction_kl_reliable_mean": reliable_kl,
        "neighbor_argmax_match_ratio": float(np.mean(teacher_assignment == neighbor_assignment)),
        "neighbor_argmax_match_ratio_reliable": reliable_acc,
        "inside_patch_match_ratio": _match_ratio(teacher_assignment, neighbor_assignment, region_masks["inside"]),
        "boundary_patch_match_ratio": _match_ratio(teacher_assignment, neighbor_assignment, region_masks["boundary"]),
        "ring_patch_match_ratio": _match_ratio(teacher_assignment, neighbor_assignment, region_masks["ring"]),
        "inside_patch_kl_mean": _masked_mean(kl_map, region_masks["inside"]),
        "boundary_patch_kl_mean": _masked_mean(kl_map, region_masks["boundary"]),
        "ring_patch_kl_mean": _masked_mean(kl_map, region_masks["ring"]),
        "same_mask_edge_consistency": edge_metrics["same_mask_edge_consistency"],
        "cross_boundary_contrast": edge_metrics["cross_boundary_contrast"],
        "boundary_preservation_score": edge_metrics["boundary_preservation_score"],
        "edge_summary": edge_summary,
        "edge_metrics": edge_metrics,
        "membership_debug": membership_debug,
        "config": cfg.to_dict(),
    }
    return PatchGraphProbeResult(
        summary=summary,
        membership=membership,
        reliable_mask=reliable_mask,
        teacher_assignment=teacher_assignment,
        neighbor_assignment=neighbor_assignment,
        teacher_probabilities=teacher,
        neighbor_probabilities=neighbor,
    )


def summarize_patch_edges(membership: np.ndarray) -> dict[str, int]:
    """Count same-mask, cross-boundary, and unlabeled 8-neighbor relations."""
    counts = {
        "num_edges": 0,
        "num_same_mask_edges": 0,
        "num_cross_boundary_edges": 0,
        "num_unlabeled_edges": 0,
    }
    height, width = membership.shape
    for y in range(height):
        for x in range(width):
            target = int(membership[y, x])
            for ny, nx in _neighbor_coords(y, x, height, width):
                if ny < y or (ny == y and nx <= x):
                    continue
                neighbor = int(membership[ny, nx])
                counts["num_edges"] += 1
                if target < 0 or neighbor < 0:
                    counts["num_unlabeled_edges"] += 1
                elif target == neighbor:
                    counts["num_same_mask_edges"] += 1
                else:
                    counts["num_cross_boundary_edges"] += 1
    return counts


def patch_region_masks(membership: np.ndarray) -> dict[str, np.ndarray]:
    """Return inside, boundary, and ring patch masks from membership ids."""
    height, width = membership.shape
    inside = np.zeros_like(membership, dtype=bool)
    boundary = np.zeros_like(membership, dtype=bool)
    ring = np.zeros_like(membership, dtype=bool)
    for y in range(height):
        for x in range(width):
            target = int(membership[y, x])
            neighbors = [int(membership[ny, nx]) for ny, nx in _neighbor_coords(y, x, height, width)]
            if target >= 0:
                if all(neighbor == target for neighbor in neighbors):
                    inside[y, x] = True
                else:
                    boundary[y, x] = True
            elif any(neighbor >= 0 for neighbor in neighbors):
                ring[y, x] = True
    return {"inside": inside, "boundary": boundary, "ring": ring}


def summarize_patch_edge_metrics(
    *,
    membership: np.ndarray,
    teacher_probabilities: np.ndarray,
    prediction_probabilities: np.ndarray,
) -> dict[str, float]:
    """Summarize edge-type behavior for boundary-focused Stage 1 diagnostics."""
    teacher_assignment = np.argmax(teacher_probabilities, axis=2)
    prediction_assignment = np.argmax(prediction_probabilities, axis=2)
    height, width = membership.shape
    same_consistency = []
    cross_contrast = []
    boundary_preservation = []
    for y in range(height):
        for x in range(width):
            target = int(membership[y, x])
            for ny, nx in _neighbor_coords(y, x, height, width):
                if ny < y or (ny == y and nx <= x):
                    continue
                neighbor = int(membership[ny, nx])
                if target >= 0 and target == neighbor:
                    same_consistency.append(float(prediction_assignment[y, x] == prediction_assignment[ny, nx]))
                elif target >= 0 and neighbor >= 0 and target != neighbor:
                    teacher_diff = teacher_assignment[y, x] != teacher_assignment[ny, nx]
                    pred_diff = prediction_assignment[y, x] != prediction_assignment[ny, nx]
                    cross_contrast.append(float(pred_diff))
                    boundary_preservation.append(float(teacher_diff == pred_diff))
    return {
        "same_mask_edge_consistency": float(np.mean(same_consistency)) if same_consistency else 0.0,
        "cross_boundary_contrast": float(np.mean(cross_contrast)) if cross_contrast else 0.0,
        "boundary_preservation_score": float(np.mean(boundary_preservation)) if boundary_preservation else 0.0,
    }


def _passes_hard_filter(score: dict[str, Any], config: PatchGraphConfig) -> bool:
    node_type = str(score.get("node_type"))
    area_ratio = float(score.get("area_ratio", 0.0))
    return (
        node_type not in set(config.hard_excluded_node_types)
        and area_ratio <= float(config.max_area_ratio)
        and area_ratio >= float(config.min_area_ratio)
    )


def _select_prototype_top_k(candidates: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    by_prototype: dict[str, list[dict[str, Any]]] = {}
    for score in candidates:
        prototype_id = str(score.get("best_prototype_id", "unknown"))
        by_prototype.setdefault(prototype_id, []).append(score)
    selected: list[dict[str, Any]] = []
    for scores in by_prototype.values():
        selected.extend(sorted(scores, key=_candidate_rank_key, reverse=True)[:top_k])
    return selected


def _select_quantile(candidates: list[dict[str, Any]], *, quantile: float) -> list[dict[str, Any]]:
    if not candidates:
        return []
    reliabilities = np.asarray([float(score.get("reliability", 0.0)) for score in candidates], dtype=np.float32)
    threshold = float(np.quantile(reliabilities, min(1.0, max(0.0, quantile))))
    return [score for score in candidates if float(score.get("reliability", 0.0)) >= threshold]


def _candidate_rank_key(score: dict[str, Any]) -> tuple[float, float, float]:
    node_type = str(score.get("node_type"))
    node_bonus = 1.0 if node_type == "valid_component" else 0.5 if node_type == "small_detail" else 0.0
    return (
        node_bonus,
        float(score.get("reliability", 0.0)),
        float(score.get("max_prototype_similarity", 0.0)),
    )


def _match_ratio(teacher_assignment: np.ndarray, prediction_assignment: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(teacher_assignment[mask] == prediction_assignment[mask]))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def _mask_id(raw_mask: dict[str, Any], index: int) -> str:
    return str(raw_mask.get("mask_id", raw_mask.get("id", index)))


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((int(size[1]), int(size[0])), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _neighbor_coords(y: int, x: int, height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width:
                coords.append((ny, nx))
    return coords


def _edge_weight(
    *,
    target_membership: int,
    neighbor_membership: int,
    neighbor_reliability: float,
    config: PatchGraphConfig,
) -> float:
    if target_membership < 0 or neighbor_membership < 0:
        return config.unlabeled_neighbor_weight
    if target_membership == neighbor_membership:
        return (1.0 + config.same_mask_bonus) * max(neighbor_reliability, config.eps)
    return config.cross_boundary_weight * max(neighbor_reliability, config.eps)


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-8)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-8)
    return left_norm @ right_norm.T


def _softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=axis, keepdims=True), 1e-8)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-9)), axis=-1)


def _kl_divergence(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.sum(
        target * (np.log(np.maximum(target, 1e-9)) - np.log(np.maximum(prediction, 1e-9))),
        axis=-1,
    )
