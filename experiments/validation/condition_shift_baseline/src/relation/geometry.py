from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


_POSITION_CORNER_PLACEMENTS = ("top_left", "top_right", "bottom_left", "bottom_right")
_EPS = 1e-12


@dataclass(frozen=True)
class PositionTransform:
    scale: float
    offset: tuple[float, float]
    placement: str
    image_size: tuple[int, int]
    shifted_size: tuple[int, int]


def position_shift_scale(params: dict[str, Any]) -> float:
    if "scale" in params:
        scale = float(params["scale"])
    else:
        center_shift_ratio = float(params.get("center_shift_ratio", params.get("max_ratio", 0.0)))
        scale = 1.0 - (2.0 * center_shift_ratio)
    return min(1.0, max(0.05, scale))


def resolve_position_shift_placement(params: dict[str, Any], seed: int) -> str:
    placement = params.get("placement", params.get("anchor", "seeded_corner"))
    if isinstance(placement, list):
        candidates = tuple(str(item) for item in placement)
        if not candidates:
            raise ValueError("position_shift placement list cannot be empty")
        return random.Random(seed).choice(candidates)
    if placement in {"seeded_corner", "random_corner"}:
        return random.Random(seed).choice(_POSITION_CORNER_PLACEMENTS)
    return str(placement)


def position_shift_offset(
    image_size: tuple[int, int],
    shifted_size: tuple[int, int],
    placement: str,
) -> tuple[int, int]:
    width, height = image_size
    shifted_width, shifted_height = shifted_size
    max_x = width - shifted_width
    max_y = height - shifted_height
    offsets = {
        "top_left": (0, 0),
        "top_right": (max_x, 0),
        "bottom_left": (0, max_y),
        "bottom_right": (max_x, max_y),
        "center": (max_x // 2, max_y // 2),
    }
    if placement not in offsets:
        raise ValueError(f"Unsupported position_shift placement={placement}")
    return offsets[placement]


def resolve_position_shift_transform(
    image_size: tuple[int, int],
    params: dict[str, Any],
    seed: int,
) -> PositionTransform:
    width, height = image_size
    scale = position_shift_scale(params)
    shifted_size = (
        max(1, min(width, int(round(width * scale)))),
        max(1, min(height, int(round(height * scale)))),
    )
    placement = resolve_position_shift_placement(params, seed)
    offset = position_shift_offset(image_size, shifted_size, placement)
    return PositionTransform(
        scale=scale,
        offset=(float(offset[0]), float(offset[1])),
        placement=placement,
        image_size=image_size,
        shifted_size=shifted_size,
    )


def as_point_array(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {array.shape}")
    if len(array) < 2:
        raise ValueError("at least two points are required for relation scores")
    return array


def apply_position_transform(points: Any, transform: PositionTransform) -> np.ndarray:
    array = as_point_array(points)
    return array * transform.scale + np.asarray(transform.offset, dtype=np.float64)


def center_points(points: Any) -> np.ndarray:
    array = as_point_array(points)
    return array - array.mean(axis=0, keepdims=True)


def mean_pairwise_distance(points: Any) -> float:
    array = as_point_array(points)
    diffs = array[:, None, :] - array[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    upper = distances[np.triu_indices(len(array), k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.mean(upper))


def normalized_centered_points(points: Any) -> np.ndarray:
    centered = center_points(points)
    scale = mean_pairwise_distance(centered)
    if scale <= _EPS:
        return centered
    return centered / scale


def absolute_position_score(reference_points: Any, query_points: Any) -> float:
    reference = as_point_array(reference_points)
    query = as_point_array(query_points)
    _validate_same_shape(reference, query)
    return float(np.mean(np.linalg.norm(query - reference, axis=1)))


def centered_position_score(reference_points: Any, query_points: Any, *, normalize_scale: bool) -> float:
    reference = normalized_centered_points(reference_points) if normalize_scale else center_points(reference_points)
    query = normalized_centered_points(query_points) if normalize_scale else center_points(query_points)
    _validate_same_shape(reference, query)
    return float(np.mean(np.linalg.norm(query - reference, axis=1)))


def pairwise_relation_score(reference_points: Any, query_points: Any, *, normalize_scale: bool) -> float:
    reference = normalized_centered_points(reference_points) if normalize_scale else center_points(reference_points)
    query = normalized_centered_points(query_points) if normalize_scale else center_points(query_points)
    _validate_same_shape(reference, query)
    ref_vectors = _upper_pairwise_vectors(reference)
    query_vectors = _upper_pairwise_vectors(query)
    return float(np.mean(np.linalg.norm(query_vectors - ref_vectors, axis=1)))


def relation_score_bundle(reference_points: Any, query_points: Any) -> dict[str, float]:
    return {
        "s_abs": absolute_position_score(reference_points, query_points),
        "s_centered_raw": centered_position_score(reference_points, query_points, normalize_scale=False),
        "s_centered_norm": centered_position_score(reference_points, query_points, normalize_scale=True),
        "s_pair_raw": pairwise_relation_score(reference_points, query_points, normalize_scale=False),
        "s_pair_norm": pairwise_relation_score(reference_points, query_points, normalize_scale=True),
    }


def _upper_pairwise_vectors(points: np.ndarray) -> np.ndarray:
    row_idx, col_idx = np.triu_indices(len(points), k=1)
    return points[col_idx] - points[row_idx]


def _validate_same_shape(reference: np.ndarray, query: np.ndarray) -> None:
    if reference.shape != query.shape:
        raise ValueError(f"point arrays must have the same shape, got {reference.shape} vs {query.shape}")
