from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from patchcore_datasets import (
    ImageFolderDataset,
    ManifestSubsetDataset,
    load_category_entries,
)
from patchcore_factory import build_patchcore, get_patchcore, make_loader
from repo_paths import REPO_ROOT, now_kst_string


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument(
        "--manifest",
        default="manifests/query_identity.jsonl",
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampler-percentage", type=float, default=0.001)
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/patchcore_identity_repro",
    )
    args = parser.parse_args()

    ns = get_patchcore()
    device = torch.device("cpu")
    output_dir = REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ns.MVTecDataset(
        str(REPO_ROOT / args.data_root),
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=ns.DatasetSplit.TRAIN,
    )
    direct_dataset = ImageFolderDataset(
        REPO_ROOT / args.data_root / args.category / "test" / "good",
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )
    manifest_entries = load_category_entries(REPO_ROOT / args.manifest, args.category)
    manifest_dataset = ManifestSubsetDataset(
        manifest_entries,
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )

    model = build_patchcore(
        device, args.imagesize, args.num_workers, args.sampler_percentage
    )
    model.fit(make_loader(train_dataset, args.batch_size, args.num_workers))

    direct_scores, _, _, _ = model.predict(
        make_loader(direct_dataset, args.batch_size, args.num_workers)
    )
    manifest_scores, _, _, _ = model.predict(
        make_loader(manifest_dataset, args.batch_size, args.num_workers)
    )

    direct_scores = [float(score) for score in direct_scores]
    manifest_scores = [float(score) for score in manifest_scores]
    per_sample = []
    abs_diffs = []

    for image_path, direct_score, manifest_score in zip(
        direct_dataset.image_paths, direct_scores, manifest_scores
    ):
        diff = manifest_score - direct_score
        abs_diff = abs(diff)
        abs_diffs.append(abs_diff)
        per_sample.append(
            {
                "source_id": image_path.name,
                "direct_score": direct_score,
                "manifest_identity_score": manifest_score,
                "diff": diff,
                "abs_diff": abs_diff,
            }
        )

    report = {
        "updated_at": now_kst_string(),
        "category": args.category,
        "manifest": str(REPO_ROOT / args.manifest),
        "sampler_percentage": args.sampler_percentage,
        "direct_clean": summarize(direct_scores),
        "manifest_identity": summarize(manifest_scores),
        "score_diff": {
            "mean_abs_diff": float(np.mean(abs_diffs)) if abs_diffs else 0.0,
            "max_abs_diff": float(np.max(abs_diffs)) if abs_diffs else 0.0,
            "allclose_atol_1e-6": bool(
                np.allclose(direct_scores, manifest_scores, atol=1e-6)
            ),
            "allclose_atol_1e-5": bool(
                np.allclose(direct_scores, manifest_scores, atol=1e-5)
            ),
        },
        "per_sample": per_sample,
    }

    output_path = output_dir / f"{args.category}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["score_diff"], ensure_ascii=True, indent=2))
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
