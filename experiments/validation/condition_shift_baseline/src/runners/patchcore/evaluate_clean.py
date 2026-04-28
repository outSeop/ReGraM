from __future__ import annotations

import sys
from pathlib import Path

_cursor = Path(__file__).resolve()
while _cursor.name != "src" and _cursor.parent != _cursor:
    _cursor = _cursor.parent
_SRC_ROOT = _cursor
_CORE_ROOT = _SRC_ROOT / "core"
for _p in (_SRC_ROOT, _CORE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contracts import build_summary, write_log, write_summary
from patchcore_factory import build_patchcore, get_patchcore, make_loader
from repo_paths import REPO_ROOT, now_kst_string


def evaluate_category(
    category: str,
    *,
    data_root: Path,
    device: torch.device,
    resize: int,
    imagesize: int,
    batch_size: int,
    num_workers: int,
    sampler_percentage: float,
) -> dict:
    ns = get_patchcore()
    train_dataset = ns.MVTecDataset(
        str(data_root),
        classname=category,
        resize=resize,
        imagesize=imagesize,
        split=ns.DatasetSplit.TRAIN,
    )
    test_dataset = ns.MVTecDataset(
        str(data_root),
        classname=category,
        resize=resize,
        imagesize=imagesize,
        split=ns.DatasetSplit.TEST,
    )

    model = build_patchcore(device, imagesize, num_workers, sampler_percentage)
    model.fit(make_loader(train_dataset, batch_size, num_workers))
    scores, segmentations, labels_gt, masks_gt = model.predict(
        make_loader(test_dataset, batch_size, num_workers)
    )

    anomaly_labels = [x[1] != "good" for x in test_dataset.data_to_iterate]
    image_auroc = ns.metrics.compute_imagewise_retrieval_metrics(scores, anomaly_labels)[
        "auroc"
    ]
    full_pixel_auroc = ns.metrics.compute_pixelwise_retrieval_metrics(
        segmentations, masks_gt
    )["auroc"]

    sel_idxs = [idx for idx, mask in enumerate(masks_gt) if np.sum(mask) > 0]
    anomaly_pixel_auroc = ns.metrics.compute_pixelwise_retrieval_metrics(
        [segmentations[i] for i in sel_idxs],
        [masks_gt[i] for i in sel_idxs],
    )["auroc"]

    return {
        "category": category,
        "train_count": len(train_dataset),
        "test_count": len(test_dataset),
        "image_auroc": float(image_auroc * 100.0),
        "full_pixel_auroc": float(full_pixel_auroc * 100.0),
        "anomaly_pixel_auroc": float(anomaly_pixel_auroc * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "breakfast_box",
            "juice_bottle",
            "pushpins",
            "screw_bag",
            "splicing_connectors",
        ],
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampler-percentage", type=float, default=0.001)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/patchcore_clean_eval.json",
    )
    parser.add_argument(
        "--log-path",
        default="experiments/validation/condition_shift_baseline/reports/patchcore_clean_eval/log.txt",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    results = []
    for category in args.categories:
        print(f"evaluating={category}", flush=True)
        result = evaluate_category(
            category,
            data_root=REPO_ROOT / args.data_root,
            device=device,
            resize=args.resize,
            imagesize=args.imagesize,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler_percentage=args.sampler_percentage,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)

    raw_results = {
        "updated_at": now_kst_string(),
        "categories": results,
        "mean_image_auroc": float(np.mean([item["image_auroc"] for item in results])),
        "mean_full_pixel_auroc": float(
            np.mean([item["full_pixel_auroc"] for item in results])
        ),
        "mean_anomaly_pixel_auroc": float(
            np.mean([item["anomaly_pixel_auroc"] for item in results])
        ),
        "sampler_percentage": args.sampler_percentage,
    }

    output_path = REPO_ROOT / args.output
    log_path = REPO_ROOT / args.log_path
    summary = build_summary(
        baseline="PatchCore",
        dataset="mvtec_loco",
        class_name="all",
        eval_type="clean",
        device=str(device),
        output_path=output_path,
        log_path=log_path,
        config={
            "data_root": str((REPO_ROOT / args.data_root).resolve()),
            "categories": args.categories,
            "resize": args.resize,
            "imagesize": args.imagesize,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "sampler_percentage": args.sampler_percentage,
            "device": args.device,
        },
        metrics={
            "mean_image_auroc": raw_results["mean_image_auroc"],
            "mean_full_pixel_auroc": raw_results["mean_full_pixel_auroc"],
            "mean_anomaly_pixel_auroc": raw_results["mean_anomaly_pixel_auroc"],
        },
        paths={
            "repo_root": REPO_ROOT,
            "data_root": (REPO_ROOT / args.data_root).resolve(),
        },
        payload=raw_results,
    )
    write_log(
        log_path,
        [
            "runner=evaluate_patchcore_clean.py",
            "baseline=PatchCore",
            "dataset=mvtec_loco",
            "eval_type=clean",
            f"categories={','.join(args.categories)}",
            f"output_path={output_path}",
        ],
    )
    write_summary(summary, output_path)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
