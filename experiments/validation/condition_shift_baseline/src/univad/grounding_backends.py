"""GroundingDINO + SAM backend adapters for shifted mask probes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def grounding_segmentation_with_backend(
    img_paths: list[str],
    save_path: str,
    config: dict[str, Any],
    *,
    backend: str,
    robustsam_root: Path | None = None,
    robustsam_checkpoint: Path | None = None,
    robustsam_model_size: str = "b",
) -> None:
    """Run UniVAD GroundingDINO segmentation with a selectable SAM backend."""
    if backend == "sam_hq":
        from models.component_segmentaion import grounding_segmentation

        grounding_segmentation(img_paths, save_path, config)
        return
    if backend != "robustsam":
        raise ValueError(f"Unsupported segmentation backend: {backend}")
    if robustsam_root is None or robustsam_checkpoint is None:
        raise ValueError("robustsam_root and robustsam_checkpoint are required for backend='robustsam'")
    grounding_segmentation_robustsam(
        img_paths,
        save_path,
        config,
        robustsam_root=Path(robustsam_root),
        robustsam_checkpoint=Path(robustsam_checkpoint),
        robustsam_model_size=robustsam_model_size,
    )


def grounding_segmentation_robustsam(
    img_paths: list[str],
    save_path: str,
    config: dict[str, Any],
    *,
    robustsam_root: Path,
    robustsam_checkpoint: Path,
    robustsam_model_size: str = "b",
) -> None:
    """Run GroundingDINO boxes and segment them with RobustSAM box prompts."""
    import cv2
    import torch
    import tqdm

    from models.component_segmentaion import (
        color_masks,
        filter_by_combine,
        merge_masks,
        turn_binary_to_int,
    )
    from models.grounded_sam import get_grounding_output, load_image, load_model

    _ensure_path(robustsam_root)
    from robust_segment_anything import sam_model_registry
    from robust_segment_anything.utils.transforms import ResizeLongestSide

    if not robustsam_checkpoint.exists():
        raise FileNotFoundError(f"RobustSAM checkpoint not found: {robustsam_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("RobustSAM shifted grounding probe requires CUDA")

    box_threshold = config["box_threshold"]
    text_threshold = config["text_threshold"]
    text_prompt = config["text_prompt"]
    dino = load_model(
        "./models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "./pretrained_ckpts/groundingdino_swint_ogc.pth",
        device,
    )
    opt = SimpleNamespace(
        model_size=robustsam_model_size,
        checkpoint_path=str(robustsam_checkpoint),
    )
    robustsam = sam_model_registry[f"vit_{robustsam_model_size}"](
        opt=opt,
        checkpoint=str(robustsam_checkpoint),
    ).to(device)
    robustsam.eval()
    sam_transform = ResizeLongestSide(robustsam.image_encoder.img_size)

    os.makedirs(save_path, exist_ok=True)
    for image_path in tqdm.tqdm(img_paths, desc="grounding_robustsam..."):
        image_pil, dino_image = load_image(image_path)
        boxes_filt, pred_phrases = get_grounding_output(
            dino,
            dino_image,
            text_prompt,
            box_threshold,
            text_threshold,
            device=device,
        )
        background_box = _background_box_indices(pred_phrases, config.get("background_prompt", ""))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        boxes_xyxy = _dino_boxes_to_xyxy(boxes_filt, image_pil.size).cpu()

        if len(boxes_xyxy) == 0:
            masks_np = []
            background = np.zeros((height, width), dtype=np.uint8)
        else:
            masks_bool = _predict_robustsam_masks(
                robustsam,
                sam_transform,
                image,
                boxes_xyxy,
                device=device,
                opt=opt,
            )
            background = _background_mask(masks_bool, background_box, shape=(height, width))
            masks_np = [
                masks_bool[index].astype(np.uint8) * 255
                for index in range(len(masks_bool))
                if index not in background_box
            ]

        if masks_np:
            if config.get("filter_by_combine", False):
                masks_np = filter_by_combine(masks_np)
            color_mask = color_masks(masks_np)
            merged_masks = merge_masks(masks_np)
        else:
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)
            merged_masks = np.zeros((height, width), dtype=np.uint8)

        image_name = "/".join((image_path.split(".")[-2]).split("/")[-3:])
        output_dir = Path(save_path) / image_name
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "grounding_mask.png"), merged_masks)
        cv2.imwrite(str(output_dir / "grounding_background.png"), turn_binary_to_int(background != 0))
        cv2.imwrite(str(output_dir / "grounding_mask_color.png"), color_mask)


def _ensure_path(path: Path) -> None:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _background_box_indices(pred_phrases: list[str], background_prompt: str) -> list[int]:
    prompts = [item for item in background_prompt.split(".") if item and item != " "]
    return [
        index
        for index, text in enumerate(pred_phrases)
        if any(prompt in text.replace(" - ", "-") for prompt in prompts)
    ]


def _dino_boxes_to_xyxy(boxes_filt, image_size: tuple[int, int]):
    import torch

    width, height = image_size
    boxes = boxes_filt.clone()
    for index in range(boxes.size(0)):
        boxes[index] = boxes[index] * torch.Tensor([width, height, width, height])
        boxes[index][:2] -= boxes[index][2:] / 2
        boxes[index][2:] += boxes[index][:2]
    return boxes


def _predict_robustsam_masks(
    model,
    transform,
    image: np.ndarray,
    boxes_xyxy,
    *,
    device: str,
    opt,
) -> np.ndarray:
    import torch

    image_t = torch.tensor(image, dtype=torch.uint8).unsqueeze(0).to(device)
    image_t = torch.permute(image_t, (0, 3, 1, 2))
    image_t_transformed = transform.apply_image_torch(image_t.float())
    boxes_t = boxes_xyxy.to(device)
    masks: list[torch.Tensor] = []
    # RobustSAM's prompt encoder follows the demo path and expects one box
    # prompt batch per image. Passing all GroundingDINO boxes at once can create
    # mismatched sparse/box embedding batch dimensions.
    with torch.no_grad():
        for box in boxes_t:
            transformed_box = transform.apply_boxes_torch(box.unsqueeze(0), image_t.shape[-2:])
            data_dict = {
                "image": image_t_transformed,
                "boxes": transformed_box.unsqueeze(0),
                "original_size": image_t.shape[-2:],
            }
            output = model.predict(opt, [data_dict], multimask_output=False, return_logits=False)[0]["masks"]
            mask = _normalize_robustsam_output_mask(output.detach().cpu())
            masks.append(mask)
    if not masks:
        return np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
    return torch.stack(masks, dim=0).numpy().astype(bool)


def _normalize_robustsam_output_mask(output) -> "torch.Tensor":
    import torch

    masks = output
    while masks.ndim > 2 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim > 2 and masks.shape[0] != 1:
        masks = masks[0]
    if masks.ndim != 2:
        masks = masks.reshape(masks.shape[-2], masks.shape[-1])
    return masks.bool()


def _background_mask(masks_bool: np.ndarray, background_box: list[int], *, shape: tuple[int, int]) -> np.ndarray:
    if not background_box:
        return np.zeros(shape, dtype=np.uint8)
    selected = [masks_bool[index] for index in background_box if index < len(masks_bool)]
    if not selected:
        return np.zeros(shape, dtype=np.uint8)
    return (np.any(np.stack(selected, axis=0), axis=0).astype(np.uint8) * 255)
