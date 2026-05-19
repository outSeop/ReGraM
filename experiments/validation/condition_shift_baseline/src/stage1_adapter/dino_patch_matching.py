"""DINO-only clean/shift patch matching diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DinoPatchMatchResult:
    """Patch-level nearest-neighbor matching between two DINO feature maps."""

    grid_shape: tuple[int, int]
    identity_similarity_map: np.ndarray
    top1_similarity_map: np.ndarray
    top1_index_map: np.ndarray
    displacement_x_map: np.ndarray
    displacement_y_map: np.ndarray
    displacement_norm_map: np.ndarray
    metrics: dict[str, float]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe summary without dense maps."""
        return {
            "grid_shape": list(self.grid_shape),
            "metrics": {key: float(value) for key, value in self.metrics.items()},
        }


def match_dino_patch_grid(reference_feature_map: np.ndarray, query_feature_map: np.ndarray) -> DinoPatchMatchResult:
    """Match each query DINO patch to the closest reference DINO patch.

    The function intentionally uses only DINO patch features. It does not use masks,
    prototypes, OT, or anomaly labels. It is meant to show how a global condition
    shift changes local patch identity and nearest-neighbor displacement.
    """
    reference = np.asarray(reference_feature_map, dtype=np.float32)
    query = np.asarray(query_feature_map, dtype=np.float32)
    if reference.shape != query.shape:
        raise ValueError(f"feature maps must have the same shape, got {reference.shape} and {query.shape}")
    if reference.ndim != 3:
        raise ValueError(f"feature maps must be HxWxD, got shape {reference.shape}")

    height, width, _ = reference.shape
    reference_flat = reference.reshape(height * width, -1)
    query_flat = query.reshape(height * width, -1)
    similarity = _cosine_matrix(query_flat, reference_flat)

    identity_indices = np.arange(height * width)
    identity_similarity = similarity[identity_indices, identity_indices]
    top1_indices = np.argmax(similarity, axis=1)
    top1_similarity = similarity[identity_indices, top1_indices]

    query_y, query_x = np.divmod(identity_indices, width)
    ref_y, ref_x = np.divmod(top1_indices, width)
    displacement_x = ref_x.astype(np.float32) - query_x.astype(np.float32)
    displacement_y = ref_y.astype(np.float32) - query_y.astype(np.float32)
    displacement_norm = np.sqrt(displacement_x**2 + displacement_y**2)
    identity_top1 = top1_indices == identity_indices
    local_top1 = (np.abs(displacement_x) <= 1.0) & (np.abs(displacement_y) <= 1.0)

    metrics = {
        "identity_similarity_mean": float(np.mean(identity_similarity)),
        "identity_similarity_median": float(np.median(identity_similarity)),
        "identity_similarity_p10": float(np.quantile(identity_similarity, 0.10)),
        "top1_similarity_mean": float(np.mean(top1_similarity)),
        "top1_similarity_median": float(np.median(top1_similarity)),
        "identity_top1_ratio": float(np.mean(identity_top1)),
        "local_3x3_top1_ratio": float(np.mean(local_top1)),
        "mean_displacement_patches": float(np.mean(displacement_norm)),
        "median_displacement_patches": float(np.median(displacement_norm)),
        "p90_displacement_patches": float(np.quantile(displacement_norm, 0.90)),
    }
    return DinoPatchMatchResult(
        grid_shape=(height, width),
        identity_similarity_map=identity_similarity.reshape(height, width),
        top1_similarity_map=top1_similarity.reshape(height, width),
        top1_index_map=top1_indices.reshape(height, width),
        displacement_x_map=displacement_x.reshape(height, width),
        displacement_y_map=displacement_y.reshape(height, width),
        displacement_norm_map=displacement_norm.reshape(height, width),
        metrics=metrics,
    )


def summarize_dino_patch_match(result: DinoPatchMatchResult) -> dict[str, Any]:
    """Return one flat row for CSV/DataFrame logging."""
    row: dict[str, Any] = {
        "grid_height": int(result.grid_shape[0]),
        "grid_width": int(result.grid_shape[1]),
    }
    row.update({key: float(value) for key, value in result.metrics.items()})
    return row


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-6)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
    return left_norm @ right_norm.T
