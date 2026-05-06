from __future__ import annotations

import math
from itertools import combinations
from typing import Any


def node_feature(node: dict[str, Any], image_size: tuple[int, int] | None = None) -> dict[str, float]:
    """Create geometry-first numeric features from a component node."""
    width, height = _image_size(image_size)
    x1, y1, x2, y2 = [float(value) for value in node.get("bbox", [0, 0, 0, 0])]
    cx, cy = [float(value) for value in node.get("centroid", [0.0, 0.0])]
    area_ratio = float(node.get("area_ratio", 0.0))
    return {
        "cx": cx / width,
        "cy": cy / height,
        "bbox_w": max(0.0, x2 - x1) / width,
        "bbox_h": max(0.0, y2 - y1) / height,
        "area_ratio": area_ratio,
    }


def edge_feature(
    node_i: dict[str, Any],
    node_j: dict[str, Any],
    image_size: tuple[int, int] | None = None,
) -> dict[str, float | str | bool]:
    """Create explicit pairwise relation features between two component nodes."""
    width, height = _image_size(image_size)
    cxi, cyi = [float(value) for value in node_i.get("centroid", [0.0, 0.0])]
    cxj, cyj = [float(value) for value in node_j.get("centroid", [0.0, 0.0])]
    dx = (cxj - cxi) / width
    dy = (cyj - cyi) / height
    bbox_i = [float(value) for value in node_i.get("bbox", [0, 0, 0, 0])]
    bbox_j = [float(value) for value in node_j.get("bbox", [0, 0, 0, 0])]
    area_i = max(float(node_i.get("area_ratio", 0.0)), 1e-12)
    area_j = max(float(node_j.get("area_ratio", 0.0)), 1e-12)
    bbox_w_i = max(bbox_i[2] - bbox_i[0], 1e-12)
    bbox_h_i = max(bbox_i[3] - bbox_i[1], 1e-12)
    bbox_w_j = max(bbox_j[2] - bbox_j[0], 1e-12)
    bbox_h_j = max(bbox_j[3] - bbox_j[1], 1e-12)
    return {
        "dx": float(dx),
        "dy": float(dy),
        "distance": float(math.sqrt(dx * dx + dy * dy)),
        "angle": float(math.atan2(dy, dx)),
        "area_ratio": float(area_j / area_i),
        "bbox_width_ratio": float(bbox_w_j / bbox_w_i),
        "bbox_height_ratio": float(bbox_h_j / bbox_h_i),
        "direction": _direction(dx, dy),
        "left": cxj < cxi,
        "right": cxj > cxi,
        "above": cyj < cyi,
        "below": cyj > cyi,
        "bbox_iou": bbox_iou(bbox_i, bbox_j),
        "contained": _contained(bbox_i, bbox_j) or _contained(bbox_j, bbox_i),
    }


def edge_features(
    nodes: list[dict[str, Any]],
    image_size: tuple[int, int] | None = None,
) -> dict[tuple[int, int], dict[str, float | str | bool]]:
    """Create features for every unordered node pair."""
    return {
        (left, right): edge_feature(nodes[left], nodes[right], image_size=image_size)
        for left, right in combinations(range(len(nodes)), 2)
    }


def node_geometry_residual(
    ref_node: dict[str, Any],
    query_node: dict[str, Any],
    image_size: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Compute matched-node residuals using geometry, not appearance."""
    ref = node_feature(ref_node, image_size=image_size)
    query = node_feature(query_node, image_size=image_size)
    centroid_shift = math.sqrt((ref["cx"] - query["cx"]) ** 2 + (ref["cy"] - query["cy"]) ** 2)
    return {
        "centroid_shift": float(centroid_shift),
        "area_ratio_change": abs(ref["area_ratio"] - query["area_ratio"]),
        "bbox_w_change": abs(ref["bbox_w"] - query["bbox_w"]),
        "bbox_h_change": abs(ref["bbox_h"] - query["bbox_h"]),
        "bbox_iou": bbox_iou(ref_node.get("bbox", [0, 0, 0, 0]), query_node.get("bbox", [0, 0, 0, 0])),
    }


def edge_residual(ref_edge: dict[str, Any], query_edge: dict[str, Any]) -> dict[str, float]:
    """Compute relation residual between matched edge features."""
    area_ref = max(float(ref_edge["area_ratio"]), 1e-12)
    area_query = max(float(query_edge["area_ratio"]), 1e-12)
    return {
        "dx": abs(float(ref_edge["dx"]) - float(query_edge["dx"])),
        "dy": abs(float(ref_edge["dy"]) - float(query_edge["dy"])),
        "distance": abs(float(ref_edge["distance"]) - float(query_edge["distance"])),
        "angle": angle_diff(float(ref_edge["angle"]), float(query_edge["angle"])),
        "log_area_ratio": abs(math.log(area_ref) - math.log(area_query)),
    }


def bbox_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Return bounding-box intersection over union."""
    ax1, ay1, ax2, ay2 = [float(value) for value in bbox_a]
    bx1, by1, bx2, by2 = [float(value) for value in bbox_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def angle_diff(angle_a: float, angle_b: float) -> float:
    """Return wrapped absolute angle difference in radians."""
    diff = (angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi
    return float(abs(diff))


def _direction(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "below" if dy >= 0 else "above"


def _contained(inner: list[float], outer: list[float]) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _image_size(image_size: tuple[int, int] | None) -> tuple[float, float]:
    if image_size is None:
        return 1.0, 1.0
    width, height = image_size
    return max(float(width), 1.0), max(float(height), 1.0)

