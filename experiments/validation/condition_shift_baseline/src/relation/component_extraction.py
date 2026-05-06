from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Component:
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int


def extract_proxy_components(
    image: Image.Image,
    *,
    max_components: int = 8,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.45,
) -> list[Component]:
    """Extract lightweight foreground components without SAM.

    This is only for the first relation sanity probe. It estimates foreground by
    distance from the border median color, then returns connected components.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background, axis=-1)
    threshold = max(float(np.percentile(distance, 80)), float(distance.mean() + 0.5 * distance.std()))
    mask = distance > threshold

    min_area = max(4, int(round(width * height * min_area_ratio)))
    max_area = max(min_area, int(round(width * height * max_area_ratio)))
    components = [
        component
        for component in _connected_components(mask)
        if min_area <= component.area <= max_area
    ]
    components.sort(key=lambda component: component.area, reverse=True)
    return components[:max_components]


def component_centroids(components: list[Component]) -> np.ndarray:
    return np.asarray([component.centroid for component in components], dtype=np.float64)


def _connected_components(mask: np.ndarray) -> list[Component]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[Component] = []

    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue
            pixels = _flood_fill(mask, seen, x, y)
            if not pixels:
                continue
            xs = np.asarray([pixel[0] for pixel in pixels], dtype=np.float64)
            ys = np.asarray([pixel[1] for pixel in pixels], dtype=np.float64)
            components.append(
                Component(
                    centroid=(float(xs.mean()), float(ys.mean())),
                    bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                    area=len(pixels),
                )
            )
    return components


def _flood_fill(mask: np.ndarray, seen: np.ndarray, start_x: int, start_y: int) -> list[tuple[int, int]]:
    height, width = mask.shape
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    seen[start_y, start_x] = True
    pixels: list[tuple[int, int]] = []

    while queue:
        x, y = queue.popleft()
        pixels.append((x, y))
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                continue
            if seen[next_y, next_x] or not mask[next_y, next_x]:
                continue
            seen[next_y, next_x] = True
            queue.append((next_x, next_y))
    return pixels
