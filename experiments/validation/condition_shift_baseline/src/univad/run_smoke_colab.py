"""Run a minimal UniVAD smoke test on Colab.

Role:
- UniVAD smoke runner
- input: prepared smoke subset and pretrained checkpoints
- output: tiny summary json and log for setup verification
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from contracts import build_summary, write_log, write_summary


def ensure_grounding_masks(univad_root: Path, smoke_root: Path, class_name: str) -> None:
    sys.path.insert(0, str(univad_root))
    from models.component_segmentaion import grounding_segmentation  # noqa: WPS433
    import yaml  # noqa: WPS433

    paths = [
        str(smoke_root / class_name / "train/good/000.png"),
        str(smoke_root / class_name / "test/good/000.png"),
        str(smoke_root / class_name / "test/logical_anomalies/000.png"),
    ]
    out_dir = univad_root / "masks" / smoke_root.name / class_name
    expected = out_dir / "test/logical_anomalies/000/grounding_mask.png"
    if expected.exists():
        return

    with (univad_root / "configs" / "class_histogram" / f"{class_name}.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    os.chdir(univad_root)
    grounding_segmentation(paths, str(out_dir), config["grounding_config"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="data/mvtec_loco_caption_smoke")
    parser.add_argument("--class-name", default="breakfast_box")
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/univad_smoke_colab/breakfast_box.json",
    )
    parser.add_argument(
        "--log-path",
        default="experiments/validation/condition_shift_baseline/reports/univad_smoke_colab/log.txt",
    )
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    smoke_root = (repo_root / args.data_root).resolve()
    univad_root = (repo_root / "external" / "UniVAD").resolve()
    output_path = (repo_root / args.output).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script on a GPU runtime.")

    os.environ.setdefault("TORCH_HOME", str(univad_root / "pretrained_ckpts" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    ensure_grounding_masks(univad_root, smoke_root, args.class_name)

    sys.path.insert(0, str(univad_root))
    os.chdir(univad_root)
    from UniVAD import UniVAD  # noqa: WPS433

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    device = "cuda"
    class_root = smoke_root / args.class_name
    train_path = class_root / "train/good/000.png"
    good_path = class_root / "test/good/000.png"
    logical_path = class_root / "test/logical_anomalies/000.png"

    model = UniVAD(image_size=args.image_size, device=device).to(device)
    normal_images = torch.stack([transform(Image.open(train_path).convert("RGB"))]).to(device)
    model.setup(
        {
            "few_shot_samples": normal_images,
            "dataset_category": args.class_name,
            "image_path": [str(train_path)],
        }
    )

    results: dict[str, object] = {
        "device": device,
        "class_name": args.class_name,
        "image_size": args.image_size,
        "samples": [],
    }

    for label, path in (("good", good_path), ("logical", logical_path)):
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        image_pil = [transform(Image.open(path).convert("RGB")).cpu().numpy()]
        pred = model(image, str(path), image_pil)
        results["samples"].append(
            {
                "label": label,
                "path": str(path),
                "pred_score": float(pred["pred_score"]),
                "pred_mask_shape": list(pred["pred_mask"].shape),
            }
        )

    log_path = (repo_root / args.log_path).resolve()
    summary = build_summary(
        baseline="UniVAD",
        dataset="mvtec_loco_smoke",
        class_name=args.class_name,
        eval_type="smoke",
        device=device,
        output_path=output_path,
        log_path=log_path,
        config={
            "data_root": str(smoke_root),
            "image_size": args.image_size,
        },
        metrics={
            "good_pred_score": next(
                (sample["pred_score"] for sample in results["samples"] if sample["label"] == "good"),
                None,
            ),
            "logical_pred_score": next(
                (sample["pred_score"] for sample in results["samples"] if sample["label"] == "logical"),
                None,
            ),
        },
        paths={
            "repo_root": repo_root,
            "data_root": smoke_root,
            "univad_root": univad_root,
        },
        payload=results,
    )
    write_log(
        log_path,
        [
            "runner=run_univad_smoke_colab.py",
            "baseline=UniVAD",
            "dataset=mvtec_loco_smoke",
            f"class_name={args.class_name}",
            "eval_type=smoke",
            f"output_path={output_path}",
        ],
    )
    write_summary(summary, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
