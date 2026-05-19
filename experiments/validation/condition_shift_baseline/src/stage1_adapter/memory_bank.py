"""Clean-normal memory bank for memory-anchored OT matching.

The memory stores clean component instances as DINO patch bags. It keeps
instance identity so count-like logical anomalies can be probed without a
query-side SAM dependency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from stage1_adapter.candidate_masks import CandidateMaskNormalizationConfig
from stage1_adapter.descriptors import CandidateDescriptor, describe_candidate_masks, descriptor_vector
from stage1_adapter.prototypes import ComponentPrototype, build_component_prototypes


@dataclass
class InstanceEntry:
    """Patch-bag representation of one clean-normal component instance."""

    instance_id: str
    source_image_id: str
    source_mask_id: str
    prototype_id: str
    patch_features: np.ndarray
    patch_positions: np.ndarray
    centroid_pos: np.ndarray
    mask_area_ratio: float
    instance_mean_feature: np.ndarray

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata without storing dense patch arrays."""
        return {
            "instance_id": self.instance_id,
            "source_image_id": self.source_image_id,
            "source_mask_id": self.source_mask_id,
            "prototype_id": self.prototype_id,
            "num_patches": int(len(self.patch_features)),
            "centroid_pos": self.centroid_pos.astype(float).tolist(),
            "mask_area_ratio": float(self.mask_area_ratio),
        }


@dataclass
class PrototypeMemoryEntry:
    """Memory patch bags and statistics for one normal role prototype."""

    prototype_id: str
    instances: list[InstanceEntry]
    centroid: np.ndarray
    within_variance: float
    within_sim_quantiles: dict[float, float]
    expected_instance_count: float
    occurrence_rate: float

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe prototype summary without dense patch arrays."""
        return {
            "prototype_id": self.prototype_id,
            "num_instances": len(self.instances),
            "feature_dim": int(self.centroid.shape[0]),
            "within_variance": float(self.within_variance),
            "within_sim_quantiles": {str(key): float(value) for key, value in self.within_sim_quantiles.items()},
            "expected_instance_count": float(self.expected_instance_count),
            "occurrence_rate": float(self.occurrence_rate),
            "instances": [instance.to_metadata_dict() for instance in self.instances],
        }


@dataclass
class SpatialGraphEdge:
    """Prototype-level relative-position Gaussian fitted from clean instances."""

    prototype_i: str
    prototype_j: str
    delta_pos_mean: np.ndarray
    delta_pos_cov: np.ndarray
    num_pairs: int

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe spatial edge summary."""
        return {
            "prototype_i": self.prototype_i,
            "prototype_j": self.prototype_j,
            "delta_pos_mean": self.delta_pos_mean.astype(float).tolist(),
            "delta_pos_cov": self.delta_pos_cov.astype(float).tolist(),
            "num_pairs": int(self.num_pairs),
        }


@dataclass
class MemoryBank:
    """Clean-normal memory bank used by memory-anchored OT matching."""

    category: str
    prototypes: dict[str, PrototypeMemoryEntry]
    spatial_graph: dict[tuple[str, str], SpatialGraphEdge]
    feature_backbone: str
    image_size: int
    num_source_images: int
    config_dict: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe memory summary without dense patch arrays."""
        return {
            "category": self.category,
            "feature_backbone": self.feature_backbone,
            "image_size": int(self.image_size),
            "num_source_images": int(self.num_source_images),
            "num_prototypes": len(self.prototypes),
            "num_spatial_edges": len(self.spatial_graph),
            "config": self.config_dict,
            "prototypes": {
                prototype_id: entry.to_metadata_dict()
                for prototype_id, entry in self.prototypes.items()
            },
            "spatial_graph": {
                _edge_key(*key): edge.to_metadata_dict()
                for key, edge in self.spatial_graph.items()
            },
        }


def build_memory_bank(
    clean_image_paths: list[Path],
    clean_mask_paths: list[Path],
    *,
    feature_extractor: Callable[[Path], np.ndarray],
    raw_masks_loader: Callable[[Path, tuple[int, int]], list[dict[str, Any]]],
    image_size: int,
    feature_backbone: str,
    max_prototypes: int = 8,
    normalization_config: CandidateMaskNormalizationConfig | None = None,
    min_pairs_for_spatial_edge: int = 3,
    category: str = "unknown",
) -> MemoryBank:
    """Build a clean-normal patch-bag memory bank.

    Args:
        clean_image_paths: Clean-normal source images.
        clean_mask_paths: Precomputed clean mask paths aligned with images.
        feature_extractor: Callable returning a DINO feature map HxWxD.
        raw_masks_loader: Callable returning normalized mask dicts for a mask path.
        image_size: DINO input size metadata.
        feature_backbone: Feature backbone metadata.
        max_prototypes: Number of clean role prototypes.
        normalization_config: Stored only for reproducibility metadata.
        min_pairs_for_spatial_edge: Minimum clean instance-pair count for an edge.
        category: Dataset category metadata.
    """
    if len(clean_image_paths) != len(clean_mask_paths):
        raise ValueError("clean_image_paths and clean_mask_paths must have the same length")
    descriptors: list[CandidateDescriptor] = []
    pending_instances: list[tuple[InstanceEntry, CandidateDescriptor]] = []
    for image_path, mask_path in zip(clean_image_paths, clean_mask_paths, strict=True):
        image = Image.open(image_path).convert("RGB")
        image_shape = (image.height, image.width)
        feature_map = feature_extractor(image_path)
        raw_masks = raw_masks_loader(mask_path, image_shape)
        image_id = image_path.name
        image_descriptors = describe_candidate_masks(
            image_id=image_id,
            feature_map=feature_map,
            raw_masks=raw_masks,
            image_shape=image_shape,
            source="clean_memory",
        )
        descriptor_by_id = {str(item.candidate_id): item for item in image_descriptors}
        descriptors.extend(image_descriptors)
        for raw_index, raw_mask in enumerate(raw_masks):
            mask_id = str(raw_mask.get("mask_id", raw_mask.get("id", raw_index)))
            descriptor = descriptor_by_id.get(mask_id)
            if descriptor is None:
                continue
            patch_features, patch_positions = _patch_bag_from_mask(
                feature_map,
                np.asarray(raw_mask["mask"], dtype=bool),
            )
            if len(patch_features) == 0:
                continue
            instance_mean = patch_features.mean(axis=0)
            pending_instances.append(
                (
                    InstanceEntry(
                        instance_id=f"{Path(image_id).stem}::mask_{mask_id}",
                        source_image_id=image_id,
                        source_mask_id=mask_id,
                        prototype_id="pending",
                        patch_features=patch_features,
                        patch_positions=patch_positions,
                        centroid_pos=patch_positions.mean(axis=0),
                        mask_area_ratio=float(descriptor.area_ratio),
                        instance_mean_feature=instance_mean,
                    ),
                    descriptor,
                )
            )

    if not pending_instances:
        raise ValueError("no clean instances were extracted for memory bank")

    component_prototypes = build_component_prototypes(descriptors, max_prototypes=max_prototypes)
    if not component_prototypes:
        raise ValueError("no component prototypes were built from clean descriptors")

    assigned_instances = _assign_instances_to_prototypes(pending_instances, component_prototypes)
    prototype_entries = _build_prototype_entries(
        assigned_instances,
        num_source_images=len(clean_image_paths),
    )
    spatial_graph = _build_spatial_graph(
        assigned_instances,
        min_pairs_for_spatial_edge=min_pairs_for_spatial_edge,
    )
    return MemoryBank(
        category=category,
        prototypes=prototype_entries,
        spatial_graph=spatial_graph,
        feature_backbone=feature_backbone,
        image_size=int(image_size),
        num_source_images=len(clean_image_paths),
        config_dict={
            "max_prototypes": max_prototypes,
            "min_pairs_for_spatial_edge": min_pairs_for_spatial_edge,
            "normalization_config": None if normalization_config is None else normalization_config.to_dict(),
        },
    )


def memory_bank_summary(memory: MemoryBank) -> dict[str, Any]:
    """Return compact W1 summary for notebook display and JSON export."""
    return {
        "category": memory.category,
        "num_source_images": memory.num_source_images,
        "num_prototypes": len(memory.prototypes),
        "num_spatial_edges": len(memory.spatial_graph),
        "prototype_instance_counts": {
            prototype_id: len(entry.instances)
            for prototype_id, entry in memory.prototypes.items()
        },
        "prototype_occurrence_rates": {
            prototype_id: float(entry.occurrence_rate)
            for prototype_id, entry in memory.prototypes.items()
        },
    }


def _assign_instances_to_prototypes(
    instances: list[tuple[InstanceEntry, CandidateDescriptor]],
    prototypes: list[ComponentPrototype],
) -> list[InstanceEntry]:
    prototype_vectors = np.stack([np.asarray(item.mean_vector, dtype=np.float32) for item in prototypes], axis=0)
    assigned = []
    for instance, descriptor in instances:
        vector = descriptor_vector(descriptor)
        similarities = _cosine_matrix(vector[None, :], prototype_vectors)[0]
        best_index = int(np.argmax(similarities))
        prototype_id = prototypes[best_index].prototype_id
        assigned.append(
            InstanceEntry(
                instance_id=instance.instance_id,
                source_image_id=instance.source_image_id,
                source_mask_id=instance.source_mask_id,
                prototype_id=prototype_id,
                patch_features=instance.patch_features,
                patch_positions=instance.patch_positions,
                centroid_pos=instance.centroid_pos,
                mask_area_ratio=instance.mask_area_ratio,
                instance_mean_feature=instance.instance_mean_feature,
            )
        )
    return assigned


def _build_prototype_entries(
    instances: list[InstanceEntry],
    *,
    num_source_images: int,
) -> dict[str, PrototypeMemoryEntry]:
    grouped: dict[str, list[InstanceEntry]] = defaultdict(list)
    for instance in instances:
        grouped[instance.prototype_id].append(instance)

    entries = {}
    for prototype_id, members in sorted(grouped.items()):
        means = np.stack([member.instance_mean_feature for member in members], axis=0).astype(np.float32)
        centroid = means.mean(axis=0)
        sims = _cosine_matrix(means, centroid[None, :]).reshape(-1)
        image_counts = Counter(member.source_image_id for member in members)
        expected_count = float(np.mean(list(image_counts.values()))) if image_counts else 0.0
        occurrence_rate = float(len(image_counts) / max(num_source_images, 1))
        entries[prototype_id] = PrototypeMemoryEntry(
            prototype_id=prototype_id,
            instances=members,
            centroid=centroid,
            within_variance=float(np.var(sims)) if len(sims) else 0.0,
            within_sim_quantiles={
                0.10: float(np.quantile(sims, 0.10)) if len(sims) else 0.0,
                0.20: float(np.quantile(sims, 0.20)) if len(sims) else 0.0,
                0.50: float(np.quantile(sims, 0.50)) if len(sims) else 0.0,
            },
            expected_instance_count=expected_count,
            occurrence_rate=occurrence_rate,
        )
    return entries


def _build_spatial_graph(
    instances: list[InstanceEntry],
    *,
    min_pairs_for_spatial_edge: int,
) -> dict[tuple[str, str], SpatialGraphEdge]:
    by_image_and_proto: dict[str, dict[str, list[InstanceEntry]]] = defaultdict(lambda: defaultdict(list))
    for instance in instances:
        by_image_and_proto[instance.source_image_id][instance.prototype_id].append(instance)

    deltas_by_pair: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    prototype_ids = sorted({instance.prototype_id for instance in instances})
    for proto_i, proto_j in combinations(prototype_ids, 2):
        for image_groups in by_image_and_proto.values():
            left_instances = image_groups.get(proto_i, [])
            right_instances = image_groups.get(proto_j, [])
            for left in left_instances:
                for right in right_instances:
                    deltas_by_pair[(proto_i, proto_j)].append(right.centroid_pos - left.centroid_pos)

    edges = {}
    for pair, deltas in sorted(deltas_by_pair.items()):
        if len(deltas) < min_pairs_for_spatial_edge:
            continue
        delta_array = np.stack(deltas, axis=0).astype(np.float32)
        cov = np.cov(delta_array.T) if len(delta_array) > 1 else np.eye(2, dtype=np.float32) * 1e-3
        cov = np.asarray(cov, dtype=np.float32).reshape(2, 2)
        cov += np.eye(2, dtype=np.float32) * 1e-6
        edges[pair] = SpatialGraphEdge(
            prototype_i=pair[0],
            prototype_j=pair[1],
            delta_pos_mean=delta_array.mean(axis=0),
            delta_pos_cov=cov,
            num_pairs=len(deltas),
        )
    return edges


def _patch_bag_from_mask(feature_map: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_shape = feature_map.shape[:2]
    patch_mask = _resize_mask(mask, feature_shape)
    ys, xs = np.where(patch_mask)
    if len(xs) == 0:
        return np.empty((0, feature_map.shape[-1]), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    features = feature_map[ys, xs].astype(np.float32)
    positions = np.stack(
        [
            xs.astype(np.float32) / max(feature_shape[1] - 1, 1),
            ys.astype(np.float32) / max(feature_shape[0] - 1, 1),
        ],
        axis=1,
    )
    return features, positions


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((int(size[1]), int(size[0])), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-6)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
    return left_norm @ right_norm.T


def _edge_key(left: str, right: str) -> str:
    return f"{left}::{right}"
