"""Rule-based mask normalization for relation graph component nodes.

Usage example:
    image = np.asarray(PIL.Image.open("image.png").convert("RGB"))
    raw_masks = [{"mask_id": "part_0", "mask": binary_mask, "score": 0.9}]
    nodes = cluster_masks_to_components(image, raw_masks, config=DEFAULT_CONFIG)
    summary = summarize_components(nodes)
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_CONFIG: dict[str, Any] = {
    "large_area_ratio": 0.01,
    "small_area_ratio": 0.003,
    "cluster_candidate_area_ratio": None,
    "min_mask_area": 20,
    "min_cluster_members": 3,
    "max_centroid_dist_ratio": 0.08,
    "max_bbox_gap_ratio": 0.03,
    "max_color_dist": 35.0,
    "min_union_area_ratio": 0.005,
    "max_spatial_dispersion_ratio": 0.12,
    "use_lab_color": True,
    "allow_mixed_labels_with_color": False,
    "absorb_nearby_singletons": False,
    "absorb_max_members": 2,
    "absorb_max_area_ratio": None,
    "absorb_max_centroid_dist_ratio": 0.12,
    "absorb_max_bbox_gap_ratio": 0.05,
}


@dataclass(frozen=True)
class MaskFeature:
    mask_id: Any
    mask: np.ndarray
    area: int
    area_ratio: float
    bbox: list[int]
    centroid: list[float]
    mean_rgb: list[float]
    std_rgb: list[float]
    mean_lab: list[float] | None
    std_lab: list[float] | None
    score: float | None
    label: str | None
    source: str
    debug: dict[str, Any]


def cluster_masks_to_components(
    image_rgb: np.ndarray,
    raw_masks: list[Any],
    *,
    config: dict[str, Any] | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize segmentation masks into relation-graph component nodes.

    This is deterministic rule-based preprocessing. It preserves large masks as
    `thing` nodes, clusters nearby small masks into `stuff_cluster` nodes when
    the cluster evidence is strong, and keeps unclustered small masks as
    `small_isolated` nodes instead of dropping them.
    """
    resolved_config = resolve_cluster_config(config, category=category)
    image = _as_rgb_array(image_rgb)
    height, width = image.shape[:2]
    image_area = int(height * width)
    lab_image = _rgb_to_lab(image) if resolved_config["use_lab_color"] else None
    records = compute_mask_features(
        image,
        raw_masks,
        resolved_config,
    )

    thing_records, small_records = split_large_small(records, resolved_config)

    nodes: list[dict[str, Any]] = []
    nodes.extend(
        _node_from_records(
            [record],
            node_type="thing",
            image=image,
            lab_image=lab_image,
            image_area=image_area,
            debug={
                "reason": _thing_reason(record, resolved_config),
                "category": category,
                "size_class": _size_class(record, resolved_config),
            },
        )
        for record in thing_records
    )

    cluster_groups = _cluster_small_records(
        small_records,
        image_shape=(height, width),
        config=resolved_config,
    )
    cluster_groups = _absorb_nearby_invalid_groups(
        cluster_groups,
        image_shape=(height, width),
        config=resolved_config,
    )
    for group, cluster_debug in cluster_groups:
        if _is_valid_stuff_cluster(group, cluster_debug, image_area=image_area, config=resolved_config):
            nodes.append(
                _node_from_records(
                    group,
                    node_type="stuff_cluster",
                    image=image,
                    lab_image=lab_image,
                    image_area=image_area,
                    debug={
                        "reason": "small_masks_clustered",
                        "category": category,
                        "cluster_conditions": cluster_debug,
                    },
                )
            )
            continue
        reason = _invalid_cluster_reason(cluster_debug, config=resolved_config)
        for record in group:
            nodes.append(
                _node_from_records(
                    [record],
                    node_type="small_isolated",
                    image=image,
                    lab_image=lab_image,
                    image_area=image_area,
                    debug={
                        "reason": reason,
                        "category": category,
                        "size_class": _size_class(record, resolved_config),
                        "failed_cluster_conditions": cluster_debug,
                    },
                )
            )

    nodes.sort(key=lambda node: (-int(node["area"]), str(node["node_type"]), str(node["mask_ids"])))
    for index, node in enumerate(nodes):
        node["node_id"] = f"node_{index:04d}"
    return nodes


def cluster_grounding_masks(
    image_rgb: np.ndarray,
    raw_masks: list[Any],
    *,
    config: dict[str, Any] | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for `cluster_masks_to_components`."""
    return cluster_masks_to_components(
        image_rgb,
        raw_masks,
        config=config,
        category=category,
    )


def _cluster_candidate_area_ratio(config: dict[str, Any]) -> float:
    value = config.get("cluster_candidate_area_ratio")
    if value is None:
        return float(config["small_area_ratio"])
    return float(value)


def resolve_cluster_config(config: dict[str, Any] | None = None, *, category: str | None = None) -> dict[str, Any]:
    resolved = dict(DEFAULT_CONFIG)
    if not config:
        return resolved
    default_overrides = config.get("default") if isinstance(config.get("default"), dict) else None
    if default_overrides:
        resolved.update(default_overrides)
    category_overrides = None
    categories = config.get("categories") if isinstance(config.get("categories"), dict) else None
    if category and categories:
        category_overrides = categories.get(category)
    if isinstance(category_overrides, dict):
        resolved.update(category_overrides)
    direct_overrides = {
        key: value
        for key, value in config.items()
        if key in DEFAULT_CONFIG
    }
    resolved.update(direct_overrides)
    return resolved


def raw_masks_from_label_image(label_image: np.ndarray | Image.Image, *, background_rgb: tuple[int, int, int] = (0, 0, 0)) -> list[dict[str, Any]]:
    """Split an RGB label image into binary raw masks, one per non-background color."""
    labels = _as_rgb_array(np.asarray(label_image.convert("RGB")) if isinstance(label_image, Image.Image) else label_image)
    flat_colors = np.unique(labels.reshape(-1, 3), axis=0)
    raw_masks: list[dict[str, Any]] = []
    for color in flat_colors:
        color_tuple = tuple(int(channel) for channel in color)
        if color_tuple == background_rgb:
            continue
        mask = np.all(labels == color, axis=-1)
        if not mask.any():
            continue
        color_id = f"rgb_{color_tuple[0]:03d}_{color_tuple[1]:03d}_{color_tuple[2]:03d}"
        raw_masks.append(
            {
                "mask_id": color_id,
                "mask": mask,
                "source": "label_image",
                "source_color_rgb": list(color_tuple),
            }
        )
    return raw_masks


def compute_mask_features(
    image_rgb: np.ndarray,
    raw_masks: list[Any],
    config: dict[str, Any] | None = None,
) -> list[MaskFeature]:
    """Compute deterministic geometric and color features for raw masks."""
    resolved_config = resolve_cluster_config(config)
    image = _as_rgb_array(image_rgb)
    image_area = int(image.shape[0] * image.shape[1])
    lab_image = _rgb_to_lab(image) if resolved_config["use_lab_color"] else None
    return _normalize_raw_masks(
        raw_masks,
        image=image,
        lab_image=lab_image,
        image_area=image_area,
        min_mask_area=int(resolved_config["min_mask_area"]),
    )


def split_large_small(
    mask_features: list[MaskFeature],
    config: dict[str, Any] | None = None,
) -> tuple[list[MaskFeature], list[MaskFeature]]:
    """Split features into graph-preserved things and small cluster candidates."""
    resolved_config = resolve_cluster_config(config)
    candidate_area_ratio = _cluster_candidate_area_ratio(resolved_config)
    thing_features: list[MaskFeature] = []
    small_features: list[MaskFeature] = []
    for feature in mask_features:
        if feature.area_ratio <= candidate_area_ratio:
            small_features.append(feature)
        else:
            thing_features.append(feature)
    return thing_features, small_features


def bbox_gap(bbox_a: list[int], bbox_b: list[int]) -> float:
    """Return Euclidean gap between two boxes; overlapping boxes have gap 0."""
    return _bbox_gap(bbox_a, bbox_b)


def color_distance(feat_a: MaskFeature, feat_b: MaskFeature, use_lab: bool = True) -> float:
    """Return feature color distance using Lab when available, otherwise RGB."""
    return _color_distance(feat_a, feat_b, use_lab_color=use_lab)


def build_small_mask_adjacency(
    small_features: list[MaskFeature],
    image_shape: tuple[int, int],
    config: dict[str, Any] | None = None,
) -> dict[int, list[int]]:
    """Build an undirected adjacency map for small mask cluster candidates."""
    resolved_config = resolve_cluster_config(config)
    height, width = image_shape
    diagonal = float((height**2 + width**2) ** 0.5)
    max_centroid_dist = float(resolved_config["max_centroid_dist_ratio"]) * diagonal
    max_bbox_gap = float(resolved_config["max_bbox_gap_ratio"]) * diagonal
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(small_features))}
    for left_index, left in enumerate(small_features):
        for right_index in range(left_index + 1, len(small_features)):
            right = small_features[right_index]
            relation = _pair_relation(
                left,
                right,
                max_centroid_dist=max_centroid_dist,
                max_bbox_gap=max_bbox_gap,
                max_color_dist=float(resolved_config["max_color_dist"]),
                use_lab_color=bool(resolved_config["use_lab_color"]),
                allow_mixed_labels_with_color=bool(resolved_config["allow_mixed_labels_with_color"]),
            )
            if relation["is_neighbor"]:
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)
    return {index: sorted(neighbors) for index, neighbors in adjacency.items()}


def connected_components(adjacency: dict[int, list[int]]) -> list[list[int]]:
    """Return connected components from an undirected adjacency mapping."""
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start_index in sorted(adjacency):
        if start_index in seen:
            continue
        stack = [start_index]
        seen.add(start_index)
        group: list[int] = []
        while stack:
            current = stack.pop()
            group.append(current)
            for next_index in adjacency.get(current, []):
                if next_index in seen:
                    continue
                seen.add(next_index)
                stack.append(next_index)
        groups.append(sorted(group))
    return groups


def build_thing_node(feat: MaskFeature, node_id: str) -> dict[str, Any]:
    """Build a JSON-serializable node for one preserved large mask."""
    return _node_from_features(
        [feat],
        node_type="thing",
        node_id=node_id,
        debug={"reason": "area_ratio>cluster_candidate_area_ratio"},
    )


def build_small_isolated_node(feat: MaskFeature, node_id: str, reason: str) -> dict[str, Any]:
    """Build a JSON-serializable node for one unclustered small mask."""
    return _node_from_features(
        [feat],
        node_type="small_isolated",
        node_id=node_id,
        debug={"reason": reason},
    )


def build_stuff_cluster_node(
    members: list[MaskFeature],
    image_shape: tuple[int, int],
    node_id: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable node from the union of clustered small masks."""
    resolved_config = resolve_cluster_config(config)
    debug = _cluster_debug(members, [], image_shape=image_shape)
    debug["reason"] = "small_masks_clustered"
    debug["valid"] = _is_valid_stuff_cluster(
        members,
        debug,
        image_area=int(image_shape[0] * image_shape[1]),
        config=resolved_config,
    )
    return _node_from_features(
        members,
        node_type="stuff_cluster",
        node_id=node_id,
        debug={"reason": "small_masks_clustered", "cluster_conditions": debug},
    )


def _normalize_raw_masks(
    raw_masks: list[Any],
    *,
    image: np.ndarray,
    lab_image: np.ndarray | None,
    image_area: int,
    min_mask_area: int,
) -> list[MaskFeature]:
    records: list[MaskFeature] = []
    for index, raw_mask in enumerate(raw_masks):
        parsed = _parse_raw_mask(raw_mask, default_mask_id=index)
        mask = np.asarray(parsed["mask"], dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"raw mask must be 2D, got shape={mask.shape}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"raw mask shape {mask.shape} does not match image shape {image.shape[:2]}")
        area = int(mask.sum())
        if area < min_mask_area:
            continue
        records.append(
            MaskFeature(
                mask_id=parsed["mask_id"],
                mask=mask,
                area=area,
                area_ratio=float(area / image_area),
                bbox=_bbox(mask),
                centroid=_centroid(mask),
                mean_rgb=_masked_mean(image, mask),
                std_rgb=_masked_std(image, mask),
                mean_lab=_masked_mean(lab_image, mask) if lab_image is not None else None,
                std_lab=_masked_std(lab_image, mask) if lab_image is not None else None,
                score=_optional_float(parsed.get("score")),
                label=_optional_str(parsed.get("label")),
                source=_optional_str(parsed.get("source")) or "raw_mask",
                debug={key: _jsonable(value) for key, value in parsed.get("debug", {}).items()},
            )
        )
    records.sort(key=lambda record: (-record.area, str(record.mask_id)))
    return records


def _parse_raw_mask(raw_mask: Any, *, default_mask_id: int) -> dict[str, Any]:
    if isinstance(raw_mask, dict):
        mask = raw_mask.get("mask")
        if mask is None:
            mask = raw_mask.get("segmentation")
        if mask is None:
            mask = raw_mask.get("binary_mask")
        if mask is None:
            raise ValueError("raw mask dict must include one of: mask, segmentation, binary_mask")
        debug = {
            "raw_source_color_rgb": raw_mask.get("source_color_rgb"),
        }
        return {
            "mask_id": raw_mask.get("mask_id", raw_mask.get("id", default_mask_id)),
            "mask": mask,
            "score": raw_mask.get("score"),
            "label": raw_mask.get("label"),
            "source": raw_mask.get("source"),
            "debug": {key: value for key, value in debug.items() if value is not None},
        }
    return {
        "mask_id": default_mask_id,
        "mask": raw_mask,
        "score": None,
        "label": None,
        "source": None,
        "debug": {},
    }


def _cluster_small_records(
    records: list[MaskFeature],
    *,
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> list[tuple[list[MaskFeature], dict[str, Any]]]:
    if not records:
        return []
    height, width = image_shape
    diagonal = float((height**2 + width**2) ** 0.5)
    max_centroid_dist = float(config["max_centroid_dist_ratio"]) * diagonal
    max_bbox_gap = float(config["max_bbox_gap_ratio"]) * diagonal
    adjacency: list[set[int]] = [set() for _ in records]
    edge_distances: dict[tuple[int, int], dict[str, float | bool | str]] = {}
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            relation = _pair_relation(
                left,
                right,
                max_centroid_dist=max_centroid_dist,
                max_bbox_gap=max_bbox_gap,
                max_color_dist=float(config["max_color_dist"]),
                use_lab_color=bool(config["use_lab_color"]),
                allow_mixed_labels_with_color=bool(config["allow_mixed_labels_with_color"]),
            )
            if relation["is_neighbor"]:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                edge_distances[(left_index, right_index)] = relation

    groups: list[tuple[list[MaskFeature], dict[str, Any]]] = []
    seen: set[int] = set()
    for start_index in range(len(records)):
        if start_index in seen:
            continue
        stack = [start_index]
        seen.add(start_index)
        group_indices: list[int] = []
        while stack:
            current = stack.pop()
            group_indices.append(current)
            for next_index in sorted(adjacency[current]):
                if next_index in seen:
                    continue
                seen.add(next_index)
                stack.append(next_index)
        group = [records[index] for index in sorted(group_indices)]
        group_edges = [
            value
            for (left_index, right_index), value in edge_distances.items()
            if left_index in group_indices and right_index in group_indices
        ]
        groups.append((group, _cluster_debug(group, group_edges, image_shape=image_shape)))
    return groups


def _absorb_nearby_invalid_groups(
    groups: list[tuple[list[MaskFeature], dict[str, Any]]],
    *,
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> list[tuple[list[MaskFeature], dict[str, Any]]]:
    if not bool(config.get("absorb_nearby_singletons", False)):
        return groups
    valid_groups: list[list[MaskFeature]] = []
    absorbed_debug: list[list[dict[str, Any]]] = []
    invalid_groups: list[tuple[list[MaskFeature], dict[str, Any]]] = []
    image_area = int(image_shape[0] * image_shape[1])
    for group, debug in groups:
        if _is_valid_stuff_cluster(group, debug, image_area=image_area, config=config):
            valid_groups.append(list(group))
            absorbed_debug.append([])
        else:
            invalid_groups.append((group, debug))
    if not valid_groups:
        return groups

    height, width = image_shape
    diagonal = float((height**2 + width**2) ** 0.5)
    max_centroid_dist = float(config["absorb_max_centroid_dist_ratio"]) * diagonal
    max_bbox_gap = float(config["absorb_max_bbox_gap_ratio"]) * diagonal
    absorb_max_area_ratio = config.get("absorb_max_area_ratio")
    if absorb_max_area_ratio is None:
        absorb_max_area_ratio = _cluster_candidate_area_ratio(config)
    remaining_invalid: list[tuple[list[MaskFeature], dict[str, Any]]] = []

    for group, debug in invalid_groups:
        if len(group) > int(config["absorb_max_members"]):
            remaining_invalid.append((group, debug))
            continue
        if any(record.area_ratio > float(absorb_max_area_ratio) for record in group):
            remaining_invalid.append((group, debug))
            continue
        best_index = None
        best_relation = None
        for valid_index, valid_group in enumerate(valid_groups):
            relation = _group_relation(group, valid_group)
            if relation["centroid_dist"] > max_centroid_dist or relation["bbox_gap"] > max_bbox_gap:
                continue
            if best_relation is None or (
                relation["bbox_gap"],
                relation["centroid_dist"],
            ) < (
                best_relation["bbox_gap"],
                best_relation["centroid_dist"],
            ):
                best_index = valid_index
                best_relation = relation
        if best_index is None or best_relation is None:
            remaining_invalid.append((group, debug))
            continue
        valid_groups[best_index].extend(group)
        absorbed_debug[best_index].append(
            {
                "absorbed_mask_ids": [_jsonable(record.mask_id) for record in group],
                "absorbed_reason": _invalid_cluster_reason(debug, config=config),
                "centroid_dist": float(best_relation["centroid_dist"]),
                "bbox_gap": float(best_relation["bbox_gap"]),
            }
        )

    absorbed_groups: list[tuple[list[MaskFeature], dict[str, Any]]] = []
    for group, absorption_events in zip(valid_groups, absorbed_debug, strict=True):
        debug = _cluster_debug(group, [], image_shape=image_shape)
        if absorption_events:
            debug["absorbed_nearby_groups"] = absorption_events
            debug["absorbed_nearby_group_count"] = len(absorption_events)
            debug["absorbed_nearby_member_count"] = sum(
                len(event["absorbed_mask_ids"]) for event in absorption_events
            )
        absorbed_groups.append((group, debug))
    return absorbed_groups + remaining_invalid


def _group_relation(left_group: list[MaskFeature], right_group: list[MaskFeature]) -> dict[str, float]:
    best_centroid_dist = float("inf")
    best_bbox_gap = float("inf")
    for left in left_group:
        for right in right_group:
            centroid_dist = _distance(left.centroid, right.centroid)
            bbox_gap = _bbox_gap(left.bbox, right.bbox)
            if (bbox_gap, centroid_dist) < (best_bbox_gap, best_centroid_dist):
                best_bbox_gap = bbox_gap
                best_centroid_dist = centroid_dist
    return {
        "centroid_dist": float(best_centroid_dist),
        "bbox_gap": float(best_bbox_gap),
    }


def _pair_relation(
    left: MaskFeature,
    right: MaskFeature,
    *,
    max_centroid_dist: float,
    max_bbox_gap: float,
    max_color_dist: float,
    use_lab_color: bool,
    allow_mixed_labels_with_color: bool,
) -> dict[str, Any]:
    centroid_dist = _distance(left.centroid, right.centroid)
    bbox_gap = _bbox_gap(left.bbox, right.bbox)
    labels_present = left.label is not None and right.label is not None
    same_label = labels_present and left.label == right.label
    color_dist = _color_distance(left, right, use_lab_color=use_lab_color)
    if same_label:
        feature_ok = True
        feature_reason = "same_label"
    elif labels_present and not allow_mixed_labels_with_color:
        feature_ok = False
        feature_reason = "different_labels"
    else:
        feature_ok = color_dist <= max_color_dist
        feature_reason = "similar_color" if feature_ok else "color_distance_too_large"
    spatial_ok = centroid_dist <= max_centroid_dist or bbox_gap <= max_bbox_gap
    return {
        "is_neighbor": bool(spatial_ok and feature_ok),
        "centroid_dist": float(centroid_dist),
        "bbox_gap": float(bbox_gap),
        "color_dist": float(color_dist),
        "same_label": bool(same_label),
        "feature_reason": feature_reason,
    }


def _cluster_debug(
    group: list[MaskFeature],
    edges: list[dict[str, Any]],
    *,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    height, width = image_shape
    image_area = int(height * width)
    union_mask = _union_mask(group)
    union_area = int(union_mask.sum())
    bbox = _bbox(union_mask)
    bbox_area = max(1, int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    color_distances = [float(edge["color_dist"]) for edge in edges]
    return {
        "num_members": len(group),
        "union_area": union_area,
        "union_area_ratio": float(union_area / image_area),
        "spatial_dispersion_ratio": _spatial_dispersion_ratio(group, image_shape=image_shape),
        "density": float(union_area / bbox_area),
        "edge_count": len(edges),
        "member_mask_ids": [_jsonable(record.mask_id) for record in group],
        "member_area_ratios": [float(record.area_ratio) for record in group],
        "member_labels": [record.label for record in group],
        "member_sources": [record.source for record in group],
        "member_color_distance_mean": float(mean(color_distances)) if color_distances else 0.0,
        "edge_feature_reasons": sorted({str(edge["feature_reason"]) for edge in edges}),
    }


def _is_valid_stuff_cluster(
    group: list[MaskFeature],
    cluster_debug: dict[str, Any],
    *,
    image_area: int,
    config: dict[str, Any],
) -> bool:
    del image_area
    return (
        len(group) >= int(config["min_cluster_members"])
        and float(cluster_debug["union_area_ratio"]) >= float(config["min_union_area_ratio"])
        and float(cluster_debug["spatial_dispersion_ratio"]) <= float(config["max_spatial_dispersion_ratio"])
    )


def _invalid_cluster_reason(cluster_debug: dict[str, Any], *, config: dict[str, Any]) -> str:
    if int(cluster_debug["num_members"]) < int(config["min_cluster_members"]):
        return "not_enough_cluster_members"
    if float(cluster_debug["union_area_ratio"]) < float(config["min_union_area_ratio"]):
        return "cluster_union_area_too_small"
    if float(cluster_debug["spatial_dispersion_ratio"]) > float(config["max_spatial_dispersion_ratio"]):
        return "cluster_spatial_dispersion_too_large"
    return "cluster_conditions_not_satisfied"


def _node_from_records(
    records: list[MaskFeature],
    *,
    node_type: str,
    image: np.ndarray,
    lab_image: np.ndarray | None,
    image_area: int,
    debug: dict[str, Any],
) -> dict[str, Any]:
    union_mask = _union_mask(records)
    member_stats = _member_stats(records, union_mask=union_mask, image_shape=image.shape[:2])
    scores = [record.score for record in records if record.score is not None]
    return {
        "node_id": "",
        "node_type": node_type,
        "source": "grounding_mask_cluster",
        "mask_ids": [_jsonable(record.mask_id) for record in records],
        "area": int(union_mask.sum()),
        "area_ratio": float(union_mask.sum() / image_area),
        "bbox": _bbox(union_mask),
        "centroid": _centroid(union_mask),
        "mean_rgb": _masked_mean(image, union_mask),
        "std_rgb": _masked_std(image, union_mask),
        "mean_lab": _masked_mean(lab_image, union_mask) if lab_image is not None else None,
        "std_lab": _masked_std(lab_image, union_mask) if lab_image is not None else None,
        "num_members": len(records),
        "member_stats": member_stats,
        "confidence": float(mean(scores)) if scores else None,
        "debug": {
            **debug,
            "raw_sources": sorted({record.source for record in records}),
            "raw_labels": sorted({record.label for record in records if record.label is not None}),
            "record_debug": [record.debug for record in records if record.debug],
        },
    }


def _node_from_features(
    records: list[MaskFeature],
    *,
    node_type: str,
    node_id: str,
    debug: dict[str, Any],
) -> dict[str, Any]:
    union_mask = _union_mask(records)
    member_stats = _member_stats(records, union_mask=union_mask, image_shape=union_mask.shape)
    scores = [record.score for record in records if record.score is not None]
    area = int(union_mask.sum())
    image_area = int(union_mask.shape[0] * union_mask.shape[1])
    return {
        "node_id": node_id,
        "node_type": node_type,
        "source": "grounding_mask_cluster",
        "mask_ids": [_jsonable(record.mask_id) for record in records],
        "area": area,
        "area_ratio": float(area / image_area),
        "bbox": _bbox(union_mask),
        "centroid": _centroid(union_mask),
        "mean_rgb": _weighted_mean([record.mean_rgb for record in records], [record.area for record in records]),
        "std_rgb": _weighted_mean([record.std_rgb for record in records], [record.area for record in records]),
        "mean_lab": _weighted_optional_mean([record.mean_lab for record in records], [record.area for record in records]),
        "std_lab": _weighted_optional_mean([record.std_lab for record in records], [record.area for record in records]),
        "num_members": len(records),
        "member_stats": member_stats,
        "confidence": float(mean(scores)) if scores else None,
        "debug": {
            **debug,
            "raw_sources": sorted({record.source for record in records}),
            "raw_labels": sorted({record.label for record in records if record.label is not None}),
            "record_debug": [record.debug for record in records if record.debug],
        },
    }


def summarize_components(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize component node counts for quick debugging and JSON reports."""
    node_type_counts: dict[str, int] = {}
    raw_mask_ids: set[str] = set()
    for node in nodes:
        node_type = str(node.get("node_type", ""))
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        raw_mask_ids.update(str(mask_id) for mask_id in node.get("mask_ids", []))
    return {
        "num_nodes": len(nodes),
        "num_thing": node_type_counts.get("thing", 0),
        "num_stuff_cluster": node_type_counts.get("stuff_cluster", 0),
        "num_small_isolated": node_type_counts.get("small_isolated", 0),
        "num_raw_masks_used": len(raw_mask_ids),
    }


def _member_stats(records: list[MaskFeature], *, union_mask: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    areas = [int(record.area) for record in records]
    centroids = [record.centroid for record in records]
    mean_rgbs = [record.mean_rgb for record in records]
    color_distances = _member_color_distances(records)
    union_bbox = _bbox(union_mask)
    bbox_area = max(1, int((union_bbox[2] - union_bbox[0]) * (union_bbox[3] - union_bbox[1])))
    return {
        "member_areas": areas,
        "member_area_mean": float(mean(areas)) if areas else 0.0,
        "member_area_std": float(np.std(np.asarray(areas, dtype=np.float64))) if areas else 0.0,
        "member_area_min": int(min(areas)) if areas else 0,
        "member_area_max": int(max(areas)) if areas else 0,
        "member_centroids": centroids,
        "member_mean_rgbs": mean_rgbs,
        "member_color_distance_mean": float(mean(color_distances)) if color_distances else 0.0,
        "spatial_dispersion": _spatial_dispersion(records),
        "spatial_dispersion_ratio": _spatial_dispersion_ratio(records, image_shape=image_shape),
        "density": float(union_mask.sum() / bbox_area),
    }


def _member_color_distances(records: list[MaskFeature]) -> list[float]:
    distances: list[float] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            distances.append(_distance(left.mean_rgb, right.mean_rgb))
    return distances


def _size_class(record: MaskFeature, config: dict[str, Any]) -> str:
    if record.area_ratio >= float(config["large_area_ratio"]):
        return "large"
    if record.area_ratio < float(config["small_area_ratio"]):
        return "small"
    return "medium_below_large_threshold"


def _thing_reason(record: MaskFeature, config: dict[str, Any]) -> str:
    if record.area_ratio >= float(config["large_area_ratio"]):
        return "area_ratio>=large_area_ratio"
    return "area_ratio>small_area_ratio_preserved_as_thing"


def _union_mask(records: list[MaskFeature]) -> np.ndarray:
    if not records:
        raise ValueError("cannot build a node from zero records")
    if len(records) == 1:
        return records[0].mask
    return np.logical_or.reduce([record.mask for record in records])


def _as_rgb_array(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape (H, W, 3), got {image.shape}")
    return image.astype(np.uint8, copy=False)


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    try:
        import cv2  # noqa: WPS433

        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    except Exception:  # noqa: BLE001
        return np.asarray(Image.fromarray(image).convert("LAB"), dtype=np.float32)


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _centroid(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def _masked_mean(image: np.ndarray | None, mask: np.ndarray) -> list[float] | None:
    if image is None:
        return None
    pixels = image[mask]
    if len(pixels) == 0:
        return [0.0, 0.0, 0.0]
    return [float(value) for value in pixels.mean(axis=0)]


def _masked_std(image: np.ndarray | None, mask: np.ndarray) -> list[float] | None:
    if image is None:
        return None
    pixels = image[mask]
    if len(pixels) == 0:
        return [0.0, 0.0, 0.0]
    return [float(value) for value in pixels.std(axis=0)]


def _distance(left: list[float], right: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def _weighted_mean(values: list[list[float]], weights: list[int]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0]
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if float(weights_array.sum()) == 0.0:
        return [float(value) for value in values_array.mean(axis=0)]
    return [float(value) for value in np.average(values_array, axis=0, weights=weights_array)]


def _weighted_optional_mean(values: list[list[float] | None], weights: list[int]) -> list[float] | None:
    present = [(value, weight) for value, weight in zip(values, weights, strict=True) if value is not None]
    if not present:
        return None
    present_values, present_weights = zip(*present, strict=True)
    return _weighted_mean(list(present_values), list(present_weights))


def _color_distance(left: MaskFeature, right: MaskFeature, *, use_lab_color: bool) -> float:
    if use_lab_color and left.mean_lab is not None and right.mean_lab is not None:
        return _distance(left.mean_lab, right.mean_lab)
    return _distance(left.mean_rgb, right.mean_rgb)


def _bbox_gap(left: list[int], right: list[int]) -> float:
    horizontal_gap = max(0, max(left[0], right[0]) - min(left[2], right[2]))
    vertical_gap = max(0, max(left[1], right[1]) - min(left[3], right[3]))
    return float((horizontal_gap**2 + vertical_gap**2) ** 0.5)


def _spatial_dispersion(records: list[MaskFeature]) -> float:
    if len(records) <= 1:
        return 0.0
    centroids = np.asarray([record.centroid for record in records], dtype=np.float64)
    center = centroids.mean(axis=0)
    return float(np.sqrt(((centroids - center) ** 2).sum(axis=1).mean()))


def _spatial_dispersion_ratio(records: list[MaskFeature], *, image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    diagonal = float((height**2 + width**2) ** 0.5)
    if diagonal == 0:
        return 0.0
    return float(_spatial_dispersion(records) / diagonal)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
