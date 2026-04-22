"""PatchCore lazy-import namespace and model/loader factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from external_loader import ensure_external_on_path


ensure_external_on_path("patchcore")


@dataclass
class PatchcoreNS:
    backbones: Any
    common: Any
    patchcore_module: Any
    sampler: Any
    metrics: Any
    DatasetSplit: Any
    MVTecDataset: Any
    IMAGENET_MEAN: Any
    IMAGENET_STD: Any


_NS: PatchcoreNS | None = None


def get_patchcore() -> PatchcoreNS:
    global _NS
    if _NS is not None:
        return _NS

    import patchcore.backbones as _backbones
    import patchcore.common as _common
    import patchcore.metrics as _metrics
    import patchcore.patchcore as _patchcore_module
    import patchcore.sampler as _sampler
    from patchcore.datasets.mvtec import (
        IMAGENET_MEAN as _IMAGENET_MEAN,
        IMAGENET_STD as _IMAGENET_STD,
        DatasetSplit as _DatasetSplit,
        MVTecDataset as _MVTecDataset,
    )

    _NS = PatchcoreNS(
        backbones=_backbones,
        common=_common,
        patchcore_module=_patchcore_module,
        sampler=_sampler,
        metrics=_metrics,
        DatasetSplit=_DatasetSplit,
        MVTecDataset=_MVTecDataset,
        IMAGENET_MEAN=_IMAGENET_MEAN,
        IMAGENET_STD=_IMAGENET_STD,
    )
    return _NS


def build_patchcore(
    device: torch.device,
    imagesize: int,
    num_workers: int,
    sampler_percentage: float,
):
    ns = get_patchcore()
    backbone = ns.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = None

    nn_method = ns.common.FaissNN(False, num_workers)
    sampler = ns.sampler.ApproximateGreedyCoresetSampler(sampler_percentage, device)

    model = ns.patchcore_module.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, imagesize, imagesize),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=sampler,
        anomaly_score_num_nn=1,
        nn_method=nn_method,
    )
    return model


def make_loader(dataset, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
