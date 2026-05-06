from __future__ import annotations

import sys
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from relation.component_extraction import Component, component_from_mask


@dataclass(frozen=True)
class SamLadComponentConfig:
    max_components: int = 12
    min_area_ratio: float = 0.0005
    small_area_ratio: float = 0.006
    max_area_ratio: float = 0.85
    iou_dedupe_threshold: float = 0.88
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.96
    box_nms_thresh: float = 0.7
    min_mask_region_area: int = 64


@dataclass(frozen=True)
class SamLadComponentResult:
    components: list[Component]
    raw_mask_count: int
    kept_mask_count: int
    merged_small_count: int
    component_source: str
    note: str = "-"


class SamLadComponentModel:
    """SAM-LAD-style component extractor using SAM automatic masks.

    The model keeps relation probe's old proxy extractor separate. It uses SAM
    masks as candidate object nodes and merges tiny masks into one coarse
    object-level component so small objects do not become noisy graph nodes.
    """

    def __init__(
        self,
        *,
        mask_generator: Any,
        config: SamLadComponentConfig | None = None,
    ) -> None:
        self.mask_generator = mask_generator
        self.config = config or SamLadComponentConfig()

    @classmethod
    def from_repo(
        cls,
        *,
        repo_root: Path,
        checkpoint_path: Path | None = None,
        sam_root: Path | None = None,
        device: str = "auto",
        config: SamLadComponentConfig | None = None,
    ) -> "SamLadComponentModel":
        resolved_config = config or SamLadComponentConfig()
        checkpoint = checkpoint_path or repo_root / "external" / "UniVAD" / "pretrained_ckpts" / "sam_hq_vit_h.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint}. "
                "Run 01 setup or place sam_hq_vit_h.pth under the shared model checkpoint cache first."
            )
        segment_anything_parent = sam_root or repo_root / "external" / "UniVAD" / "models"
        if segment_anything_parent.name == "segment_anything":
            segment_anything_parent = segment_anything_parent.parent
        if not (segment_anything_parent / "segment_anything").exists():
            raise FileNotFoundError(
                f"segment_anything package not found under: {segment_anything_parent}. "
                "Run 01 UniVAD setup so recursive submodules are available."
            )
        if str(segment_anything_parent) not in sys.path:
            sys.path.insert(0, str(segment_anything_parent))

        ensure_numpy_1_for_torch()
        import torch  # noqa: WPS433
        from segment_anything import SamAutomaticMaskGenerator, sam_hq_model_registry  # noqa: WPS433

        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        sam = sam_hq_model_registry["vit_h"](checkpoint=str(checkpoint)).to(resolved_device)
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=resolved_config.points_per_side,
            points_per_batch=16,
            pred_iou_thresh=resolved_config.pred_iou_thresh,
            stability_score_thresh=resolved_config.stability_score_thresh,
            stability_score_offset=1.0,
            box_nms_thresh=resolved_config.box_nms_thresh,
            crop_n_layers=1,
            crop_nms_thresh=0.7,
            crop_overlap_ratio=512 / 1500,
            crop_n_points_downscale_factor=1,
            point_grids=None,
            min_mask_region_area=resolved_config.min_mask_region_area,
            output_mode="binary_mask",
        )
        return cls(mask_generator=mask_generator, config=resolved_config)

    def extract(self, image: Image.Image) -> SamLadComponentResult:
        rgb = np.asarray(image.convert("RGB"))
        raw_masks = self.mask_generator.generate(rgb)
        return extract_sam_lad_components_from_masks(
            raw_masks,
            image_size=image.size,
            config=self.config,
        )


def extract_sam_lad_components_from_masks(
    raw_masks: list[Any],
    *,
    image_size: tuple[int, int],
    config: SamLadComponentConfig | None = None,
) -> SamLadComponentResult:
    resolved_config = config or SamLadComponentConfig()
    image_area = int(image_size[0] * image_size[1])
    min_area = max(1, int(round(image_area * resolved_config.min_area_ratio)))
    small_area = max(min_area, int(round(image_area * resolved_config.small_area_ratio)))
    max_area = max(small_area, int(round(image_area * resolved_config.max_area_ratio)))

    candidates = _extract_binary_masks(raw_masks)
    selected_masks: list[np.ndarray] = []
    small_masks: list[np.ndarray] = []
    dropped_large = 0
    for mask in sorted(candidates, key=lambda item: int(item.sum()), reverse=True):
        area = int(mask.sum())
        if area < min_area:
            continue
        if area > max_area:
            dropped_large += 1
            continue
        if area < small_area:
            small_masks.append(mask)
            continue
        if _overlaps_existing(mask, selected_masks, resolved_config.iou_dedupe_threshold):
            continue
        selected_masks.append(mask)

    components = [
        component
        for mask in selected_masks
        if (component := component_from_mask(mask, source="sam_lad")) is not None
    ]
    if small_masks:
        merged_small = np.logical_or.reduce(small_masks)
        if not _overlaps_existing(merged_small, selected_masks, resolved_config.iou_dedupe_threshold):
            component = component_from_mask(merged_small, source="sam_lad_merged_small")
            if component is not None:
                components.append(component)

    if not components and candidates:
        union_mask = np.logical_or.reduce(candidates)
        component = component_from_mask(union_mask, source="sam_lad_object_fallback")
        if component is not None:
            components.append(component)

    components.sort(key=lambda component: component.area, reverse=True)
    components = components[: resolved_config.max_components]
    note_parts = []
    if small_masks:
        note_parts.append(f"merged_small_masks={len(small_masks)}")
    if dropped_large:
        note_parts.append(f"dropped_large_masks={dropped_large}")
    if components and components[0].source == "sam_lad_object_fallback":
        note_parts.append("object_fallback")
    return SamLadComponentResult(
        components=components,
        raw_mask_count=len(raw_masks),
        kept_mask_count=len(components),
        merged_small_count=len(small_masks),
        component_source="sam_lad",
        note="; ".join(note_parts) if note_parts else "-",
    )


def ensure_numpy_1_for_torch() -> None:
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        return
    if numpy_version.split(".", maxsplit=1)[0] == "1":
        return
    raise RuntimeError(
        "SAM-LAD component extraction needs a torch/SAM runtime compatible with NumPy 1.x. "
        f"Current numpy={numpy_version}. In Colab, run the 05_sam_lad_relation_probe "
        "runtime guard cell to install numpy==1.26.4, restart the runtime, then rerun."
    )


def _extract_binary_masks(raw_masks: list[Any]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for raw_mask in raw_masks:
        segmentation = raw_mask.get("segmentation") if isinstance(raw_mask, dict) else raw_mask
        mask = np.asarray(segmentation, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            continue
        masks.extend(_split_connected_masks(mask))
    return masks


def _split_connected_masks(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue
            pixels = _flood_fill(mask, seen, x, y)
            component = np.zeros((height, width), dtype=bool)
            if pixels:
                ys = [pixel[1] for pixel in pixels]
                xs = [pixel[0] for pixel in pixels]
                component[ys, xs] = True
                components.append(component)
    return components


def _flood_fill(mask: np.ndarray, seen: np.ndarray, start_x: int, start_y: int) -> list[tuple[int, int]]:
    height, width = mask.shape
    stack = [(start_x, start_y)]
    seen[start_y, start_x] = True
    pixels: list[tuple[int, int]] = []
    while stack:
        x, y = stack.pop()
        pixels.append((x, y))
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                continue
            if seen[next_y, next_x] or not mask[next_y, next_x]:
                continue
            seen[next_y, next_x] = True
            stack.append((next_x, next_y))
    return pixels


def _overlaps_existing(mask: np.ndarray, selected_masks: list[np.ndarray], threshold: float) -> bool:
    return any(_mask_iou(mask, selected) >= threshold for selected in selected_masks)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = float(np.logical_and(left, right).sum())
    union = float(np.logical_or(left, right).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union
