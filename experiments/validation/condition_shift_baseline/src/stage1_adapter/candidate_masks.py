"""Candidate mask normalization before Stage 1 descriptor extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateMaskNormalizationConfig:
    """Rule-based size normalization for raw segmentation proposals."""

    max_mask_area_ratio: float = 0.30
    min_mask_area_ratio: float = 0.001
    small_cluster_area_ratio: float = 0.006
    min_cluster_members: int = 3
    min_cluster_union_area_ratio: float = 0.004
    max_cluster_union_area_ratio: float = 0.25
    max_centroid_dist_ratio: float = 0.10
    max_bbox_gap_ratio: float = 0.04

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_candidate_masks(
    raw_masks: list[dict[str, Any]],
    *,
    image_shape: tuple[int, int],
    config: CandidateMaskNormalizationConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove oversized proposals and merge small nearby masks into stuff candidates.

    The output remains a raw-mask list so downstream descriptor/prototype code
    can run unchanged. Isolated tiny masks are excluded from Stage 1 because
    they are usually unstable proposal fragments; their ids are still kept in
    the debug summary for auditability.
    """
    cfg = config or CandidateMaskNormalizationConfig()
    records = [_mask_record(raw_mask, index, image_shape=image_shape) for index, raw_mask in enumerate(raw_masks)]
    keep_records = []
    small_records = []
    excluded_large = []
    for record in records:
        if record["area_ratio"] > cfg.max_mask_area_ratio:
            excluded_large.append(record)
        elif record["area_ratio"] <= cfg.small_cluster_area_ratio:
            small_records.append(record)
        else:
            keep_records.append(record)

    normalized: list[dict[str, Any]] = [_record_to_mask(record, normalization_type="thing_candidate") for record in keep_records]
    cluster_groups = _connected_small_groups(small_records, image_shape=image_shape, config=cfg)
    excluded_small = []
    for group in cluster_groups:
        union_mask = np.logical_or.reduce([record["mask"] for record in group])
        union_area_ratio = float(union_mask.sum() / max(1, image_shape[0] * image_shape[1]))
        if (
            len(group) >= cfg.min_cluster_members
            and union_area_ratio >= cfg.min_cluster_union_area_ratio
            and union_area_ratio <= cfg.max_cluster_union_area_ratio
        ):
            normalized.append(_cluster_to_mask(group, union_mask=union_mask, union_area_ratio=union_area_ratio))
        else:
            excluded_small.extend(group)

    normalized.sort(key=lambda item: (-int(np.asarray(item["mask"], dtype=bool).sum()), str(item["mask_id"])))
    summary = {
        "num_input_masks": len(raw_masks),
        "num_output_masks": len(normalized),
        "num_thing_candidates": len(keep_records),
        "num_stuff_clusters": sum(str(item.get("normalization_type")) == "stuff_cluster_candidate" for item in normalized),
        "num_excluded_large": len(excluded_large),
        "num_excluded_small": len(excluded_small),
        "excluded_large_ids": [record["mask_id"] for record in excluded_large],
        "excluded_small_ids": [record["mask_id"] for record in excluded_small],
        "config": cfg.to_dict(),
    }
    return normalized, summary


def _mask_record(raw_mask: dict[str, Any], index: int, *, image_shape: tuple[int, int]) -> dict[str, Any]:
    mask = np.asarray(raw_mask["mask"], dtype=bool)
    area = int(mask.sum())
    image_area = max(1, image_shape[0] * image_shape[1])
    return {
        "mask_id": str(raw_mask.get("mask_id", raw_mask.get("id", index))),
        "mask": mask,
        "area": area,
        "area_ratio": float(area / image_area),
        "bbox": _bbox(mask),
        "centroid": _centroid(mask),
        "raw": raw_mask,
    }


def _record_to_mask(record: dict[str, Any], *, normalization_type: str) -> dict[str, Any]:
    raw = dict(record["raw"])
    raw["mask_id"] = record["mask_id"]
    raw["mask"] = record["mask"]
    raw["normalization_type"] = normalization_type
    raw["member_mask_ids"] = [record["mask_id"]]
    raw["debug"] = {
        **dict(raw.get("debug", {})),
        "stage1_candidate_normalization": {
            "area_ratio": record["area_ratio"],
            "normalization_type": normalization_type,
        },
    }
    return raw


def _cluster_to_mask(group: list[dict[str, Any]], *, union_mask: np.ndarray, union_area_ratio: float) -> dict[str, Any]:
    member_ids = [record["mask_id"] for record in group]
    return {
        "mask_id": "stuff_cluster::" + "::".join(member_ids[:5]) + (f"::n{len(member_ids)}" if len(member_ids) > 5 else ""),
        "mask": union_mask,
        "source": "stage1_candidate_normalization",
        "normalization_type": "stuff_cluster_candidate",
        "member_mask_ids": member_ids,
        "debug": {
            "stage1_candidate_normalization": {
                "normalization_type": "stuff_cluster_candidate",
                "member_mask_ids": member_ids,
                "member_area_ratios": [record["area_ratio"] for record in group],
                "union_area_ratio": union_area_ratio,
            },
        },
    }


def _connected_small_groups(
    records: list[dict[str, Any]],
    *,
    image_shape: tuple[int, int],
    config: CandidateMaskNormalizationConfig,
) -> list[list[dict[str, Any]]]:
    if not records:
        return []
    diagonal = float((image_shape[0] ** 2 + image_shape[1] ** 2) ** 0.5)
    adjacency = {index: [] for index in range(len(records))}
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            centroid_dist = float(np.linalg.norm(np.asarray(left["centroid"]) - np.asarray(right["centroid"])))
            gap = _bbox_gap(left["bbox"], right["bbox"])
            if (
                centroid_dist <= config.max_centroid_dist_ratio * diagonal
                or gap <= config.max_bbox_gap_ratio * diagonal
            ):
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)
    groups = []
    seen = set()
    for index in range(len(records)):
        if index in seen:
            continue
        stack = [index]
        component = []
        seen.add(index)
        while stack:
            current = stack.pop()
            component.append(records[current])
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append(component)
    return groups


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _centroid(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def _bbox_gap(left: list[int], right: list[int]) -> float:
    horizontal = max(0, max(left[0], right[0]) - min(left[2], right[2]))
    vertical = max(0, max(left[1], right[1]) - min(left[3], right[3]))
    return float((horizontal**2 + vertical**2) ** 0.5)
