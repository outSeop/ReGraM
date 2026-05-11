"""Feature descriptors for domain-specific component proposal adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CandidateDescriptor:
    """JSON-serializable descriptor for one candidate component mask."""

    candidate_id: str
    source_image_id: str
    source: str
    area: int
    area_ratio: float
    bbox: list[int]
    centroid: list[float]
    aspect_ratio: float
    fill_ratio: float
    compactness: float
    inside_mean: list[float]
    inside_std: list[float]
    inside_feature_coherence: float
    ring_mean: list[float]
    ring_std: list[float]
    ring_feature_distance: float
    neighbor_count: int
    contained_neighbor_count: int
    overlap_neighbor_count: int
    nearest_neighbor_dist_ratio: float | None
    debug: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_candidate_masks(
    *,
    image_id: str,
    feature_map: np.ndarray,
    raw_masks: list[dict[str, Any]],
    image_shape: tuple[int, int],
    source: str,
    ring_radius: int = 2,
) -> list[CandidateDescriptor]:
    """Describe raw candidate masks using geometry, inside features, and context."""
    if feature_map.ndim != 3:
        raise ValueError(f"feature_map must be HxWxC, got shape={feature_map.shape}")
    height, width = image_shape
    image_area = int(height * width)
    parsed_masks = [_parse_raw_mask(raw_mask, index) for index, raw_mask in enumerate(raw_masks)]
    descriptors: list[CandidateDescriptor] = []
    for index, raw_mask in enumerate(parsed_masks):
        mask = raw_mask["mask"]
        area = int(mask.sum())
        if area <= 0:
            continue
        bbox = _bbox(mask)
        feature_mask = _resize_mask(mask, feature_map.shape[:2])
        ring_mask = _ring_mask(feature_mask, radius=ring_radius)
        inside_values = _masked_features(feature_map, feature_mask)
        ring_values = _masked_features(feature_map, ring_mask)
        inside_mean, inside_std, coherence = _feature_stats(inside_values)
        ring_mean, ring_std, _ = _feature_stats(ring_values)
        neighbor_stats = _neighbor_stats(index, parsed_masks, image_shape=(height, width))
        descriptors.append(
            CandidateDescriptor(
                candidate_id=str(raw_mask["mask_id"]),
                source_image_id=image_id,
                source=source,
                area=area,
                area_ratio=float(area / image_area),
                bbox=bbox,
                centroid=_centroid(mask),
                aspect_ratio=_aspect_ratio(bbox),
                fill_ratio=_fill_ratio(mask, bbox),
                compactness=_compactness(mask),
                inside_mean=inside_mean.tolist(),
                inside_std=inside_std.tolist(),
                inside_feature_coherence=float(coherence),
                ring_mean=ring_mean.tolist(),
                ring_std=ring_std.tolist(),
                ring_feature_distance=float(np.linalg.norm(inside_mean - ring_mean)),
                neighbor_count=neighbor_stats["neighbor_count"],
                contained_neighbor_count=neighbor_stats["contained_neighbor_count"],
                overlap_neighbor_count=neighbor_stats["overlap_neighbor_count"],
                nearest_neighbor_dist_ratio=neighbor_stats["nearest_neighbor_dist_ratio"],
                debug={
                    "source_color_rgb": raw_mask.get("source_color_rgb"),
                    "feature_grid_shape": list(feature_map.shape[:2]),
                    "ring_radius": ring_radius,
                },
            )
        )
    return descriptors


def descriptor_vector(descriptor: CandidateDescriptor, *, include_context: bool = True) -> np.ndarray:
    """Return the vector used for prototype matching."""
    geometry = np.asarray(
        [
            descriptor.area_ratio,
            descriptor.centroid[0],
            descriptor.centroid[1],
            descriptor.aspect_ratio,
            descriptor.fill_ratio,
            descriptor.compactness,
        ],
        dtype=np.float32,
    )
    inside = np.asarray(descriptor.inside_mean, dtype=np.float32)
    if not include_context:
        return np.concatenate([geometry, inside])
    ring = np.asarray(descriptor.ring_mean, dtype=np.float32)
    context = np.asarray(
        [
            descriptor.inside_feature_coherence,
            descriptor.ring_feature_distance,
            float(descriptor.neighbor_count),
            float(descriptor.contained_neighbor_count),
            float(descriptor.overlap_neighbor_count),
        ],
        dtype=np.float32,
    )
    return np.concatenate([geometry, inside, ring, context])


def _parse_raw_mask(raw_mask: dict[str, Any], index: int) -> dict[str, Any]:
    mask = raw_mask.get("mask")
    if mask is None:
        mask = raw_mask.get("segmentation")
    if mask is None:
        mask = raw_mask.get("binary_mask")
    if mask is None:
        raise ValueError("raw mask must contain mask, segmentation, or binary_mask")
    parsed = dict(raw_mask)
    parsed["mask"] = np.asarray(mask, dtype=bool)
    parsed["mask_id"] = raw_mask.get("mask_id", raw_mask.get("id", index))
    return parsed


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((int(size[1]), int(size[0])), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _ring_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    dilated = mask.copy()
    for _ in range(max(1, int(radius))):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[1:-1, 1:-1],
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[:-2, :-2],
            padded[:-2, 2:],
            padded[2:, :-2],
            padded[2:, 2:],
        ]
        dilated = np.logical_or.reduce(neighbors)
    return np.logical_and(dilated, ~mask)


def _masked_features(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = feature_map[mask]
    if values.size == 0:
        return feature_map.reshape(-1, feature_map.shape[-1])
    return values.reshape(-1, feature_map.shape[-1])


def _feature_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = values.astype(np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    centered = values - mean[None, :]
    distances = np.linalg.norm(centered, axis=1)
    mean_norm = float(np.linalg.norm(mean))
    coherence = 1.0 / (1.0 + float(distances.mean()) / max(mean_norm, 1e-6))
    return mean, std, coherence


def _neighbor_stats(index: int, raw_masks: list[dict[str, Any]], *, image_shape: tuple[int, int]) -> dict[str, Any]:
    target = raw_masks[index]["mask"]
    target_area = max(1, int(target.sum()))
    target_centroid = np.asarray(_centroid(target), dtype=np.float32)
    diagonal = float((image_shape[0] ** 2 + image_shape[1] ** 2) ** 0.5)
    neighbor_count = 0
    contained_neighbor_count = 0
    overlap_neighbor_count = 0
    nearest_dist = None
    for other_index, other in enumerate(raw_masks):
        if other_index == index:
            continue
        other_mask = other["mask"]
        other_area = max(1, int(other_mask.sum()))
        intersection = int(np.logical_and(target, other_mask).sum())
        if intersection > 0:
            overlap_neighbor_count += 1
        if intersection / min(target_area, other_area) > 0.8 and other_area < target_area:
            contained_neighbor_count += 1
        dist = float(np.linalg.norm(target_centroid - np.asarray(_centroid(other_mask), dtype=np.float32)))
        nearest_dist = dist if nearest_dist is None else min(nearest_dist, dist)
        if intersection > 0 or dist / max(diagonal, 1e-6) < 0.15:
            neighbor_count += 1
    return {
        "neighbor_count": neighbor_count,
        "contained_neighbor_count": contained_neighbor_count,
        "overlap_neighbor_count": overlap_neighbor_count,
        "nearest_neighbor_dist_ratio": None if nearest_dist is None else float(nearest_dist / max(diagonal, 1e-6)),
    }


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _centroid(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0]
    height, width = mask.shape
    return [float(xs.mean() / max(width, 1)), float(ys.mean() / max(height, 1))]


def _aspect_ratio(bbox: list[int]) -> float:
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    return float(width / height)


def _fill_ratio(mask: np.ndarray, bbox: list[int]) -> float:
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    return float(mask.sum() / max(1, width * height))


def _compactness(mask: np.ndarray) -> float:
    bbox = _bbox(mask)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    extent = max(width, height)
    return float(mask.sum() / max(1, extent * extent))
