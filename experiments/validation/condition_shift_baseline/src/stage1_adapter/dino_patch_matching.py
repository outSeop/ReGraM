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


@dataclass(frozen=True)
class DinoRepresentationShiftResult:
    """Same-location DINO feature displacement diagnostics."""

    grid_shape: tuple[int, int]
    delta_norm_map: np.ndarray
    delta_cosine_to_mean_map: np.ndarray
    clean_pca_xy: np.ndarray
    query_pca_xy: np.ndarray
    delta_pca_xy: np.ndarray
    pca_explained_variance: np.ndarray
    delta_pca_explained_variance: np.ndarray
    metrics: dict[str, float]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe summary without dense arrays."""
        return {
            "grid_shape": list(self.grid_shape),
            "pca_explained_variance": self.pca_explained_variance.astype(float).tolist(),
            "delta_pca_explained_variance": self.delta_pca_explained_variance.astype(float).tolist(),
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


def analyze_dino_representation_shift(
    reference_feature_map: np.ndarray,
    query_feature_map: np.ndarray,
    *,
    n_components: int = 3,
) -> DinoRepresentationShiftResult:
    """Analyze same-location feature displacement in DINO representation space."""
    reference = np.asarray(reference_feature_map, dtype=np.float32)
    query = np.asarray(query_feature_map, dtype=np.float32)
    if reference.shape != query.shape:
        raise ValueError(f"feature maps must have the same shape, got {reference.shape} and {query.shape}")
    if reference.ndim != 3:
        raise ValueError(f"feature maps must be HxWxD, got shape {reference.shape}")

    height, width, dim = reference.shape
    reference_flat = reference.reshape(height * width, dim)
    query_flat = query.reshape(height * width, dim)
    delta = query_flat - reference_flat
    mean_delta = delta.mean(axis=0)
    delta_norm = np.linalg.norm(delta, axis=1)
    mean_delta_norm = float(np.linalg.norm(mean_delta))
    delta_cosine_to_mean = _cosine_to_vector(delta, mean_delta)
    delta_unit = delta / np.maximum(delta_norm[:, None], 1e-6)
    pairwise_cosine_mean = _mean_pairwise_cosine(delta_unit)
    residual_after_mean = np.linalg.norm(delta - mean_delta[None, :], axis=1)

    combined_pca_xy, explained = _pca_transform(
        np.concatenate([reference_flat, query_flat], axis=0),
        n_components=max(2, n_components),
    )
    clean_pca_xy = combined_pca_xy[: len(reference_flat), :2]
    query_pca_xy = combined_pca_xy[len(reference_flat) :, :2]
    delta_pca, delta_explained = _pca_transform(delta, n_components=max(2, n_components))

    metrics = {
        "mean_delta_norm": float(np.mean(delta_norm)),
        "median_delta_norm": float(np.median(delta_norm)),
        "p90_delta_norm": float(np.quantile(delta_norm, 0.90)),
        "mean_delta_vector_norm": mean_delta_norm,
        "delta_direction_consistency_mean": float(np.mean(delta_cosine_to_mean)),
        "delta_direction_consistency_median": float(np.median(delta_cosine_to_mean)),
        "pairwise_delta_cosine_mean": float(pairwise_cosine_mean),
        "residual_after_mean_shift_mean": float(np.mean(residual_after_mean)),
        "residual_after_mean_shift_median": float(np.median(residual_after_mean)),
        "delta_pc1_explained": float(delta_explained[0]) if len(delta_explained) > 0 else 0.0,
        "delta_pc2_explained": float(delta_explained[1]) if len(delta_explained) > 1 else 0.0,
        "delta_pc3_explained": float(delta_explained[2]) if len(delta_explained) > 2 else 0.0,
        "delta_pc3_cumulative": float(np.sum(delta_explained[:3])),
    }
    return DinoRepresentationShiftResult(
        grid_shape=(height, width),
        delta_norm_map=delta_norm.reshape(height, width),
        delta_cosine_to_mean_map=delta_cosine_to_mean.reshape(height, width),
        clean_pca_xy=clean_pca_xy.astype(np.float32),
        query_pca_xy=query_pca_xy.astype(np.float32),
        delta_pca_xy=delta_pca[:, :2].astype(np.float32),
        pca_explained_variance=explained.astype(np.float32),
        delta_pca_explained_variance=delta_explained.astype(np.float32),
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


def summarize_dino_representation_shift(result: DinoRepresentationShiftResult) -> dict[str, Any]:
    """Return one flat row for representation-shift logging."""
    row: dict[str, Any] = {
        "grid_height": int(result.grid_shape[0]),
        "grid_width": int(result.grid_shape[1]),
    }
    row.update({key: float(value) for key, value in result.metrics.items()})
    return row


def _cosine_to_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-6:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    matrix_norm = np.maximum(np.linalg.norm(matrix, axis=1), 1e-6)
    return ((matrix @ vector) / (matrix_norm * vector_norm)).astype(np.float32)


def _mean_pairwise_cosine(unit_vectors: np.ndarray) -> float:
    n = int(len(unit_vectors))
    if n <= 1:
        return 0.0
    summed = unit_vectors.sum(axis=0)
    # Sum over off-diagonal dot products divided by the number of ordered pairs.
    return float((float(summed @ summed) - n) / max(n * (n - 1), 1))


def _pca_transform(matrix: np.ndarray, *, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix.astype(np.float32) - matrix.astype(np.float32).mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    transformed = centered @ components
    variance = singular_values**2
    total = float(np.sum(variance))
    explained = variance[:n_components] / max(total, 1e-8)
    return transformed.astype(np.float32), explained.astype(np.float32)


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-6)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
    return left_norm @ right_norm.T
