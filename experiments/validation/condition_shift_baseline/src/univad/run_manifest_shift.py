"""Run UniVAD condition-shift evaluation from a manifest.

Role:
- UniVAD sibling runner to run_patchcore_manifest_shift.py
- input: manifest jsonl, category, raw LOCO dataset root
- output: summary json and log, shape-compatible with PatchCore runner's
  condition_shift_baseline.summary.v1 schema

Notes:
- UniVAD requires CUDA in practice; this runner raises otherwise.
- os.chdir(univad_root) is scoped to a contextmanager so the process CWD is
  restored after the run (relevant when this runner is invoked from a notebook
  or composed with other baselines in the same process).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from augmentation_runtime import apply_augmentation, build_manifest_entries, load_manifest
from contracts import build_summary, write_log, write_summary
from external_loader import ensure_external_on_path
from patchcore_datasets import resolve_manifest_image_path
from preview_utils import build_preview_images
from repo_paths import REPO_ROOT, finish_phase, log_phase, now_kst_string
from wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_preview_images_to_wandb,
    log_summary_to_wandb,
)


@contextlib.contextmanager
def univad_runtime_context(univad_root: Path):
    previous_cwd = Path.cwd()
    os.environ.setdefault("TORCH_HOME", str(univad_root / "pretrained_ckpts" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.chdir(univad_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def pil_to_univad_inputs(image: Image.Image, transform, device: torch.device):
    tensor = transform(image)
    batched = tensor.unsqueeze(0).to(device)
    image_pil_list = [tensor.cpu().numpy()]
    return batched, image_pil_list


def score_image(model, image: Image.Image, image_path: str, transform, device) -> float:
    batched, image_pil_list = pil_to_univad_inputs(image, transform, device)
    with torch.no_grad():
        pred = model(batched, image_path, image_pil_list)
    return float(pred["pred_score"].item() if torch.is_tensor(pred["pred_score"]) else pred["pred_score"])


def load_reference_tensors(
    train_good_dir: Path, *, k_shot: int, round_idx: int, transform, device
) -> tuple[torch.Tensor, list[str]]:
    paths = sorted(train_good_dir.glob("*.png"))
    if len(paths) < round_idx + k_shot:
        raise RuntimeError(
            f"Not enough train/good images in {train_good_dir} for k_shot={k_shot}, round={round_idx}. "
            f"Available={len(paths)}."
        )
    selected = paths[round_idx : round_idx + k_shot]
    tensors = torch.cat(
        [transform(Image.open(p).convert("RGB")).unsqueeze(0) for p in selected], dim=0
    ).to(device)
    return tensors, [str(p) for p in selected]


def summarize_scores(scores: list[float], threshold: float) -> dict:
    array = np.asarray(scores, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "fpr_over_clean_max": float((array > threshold).mean()) if array.size else 0.0,
    }


def compute_image_auroc(normal_scores: list[float], anomaly_scores: list[float]) -> float:
    if not normal_scores or not anomaly_scores:
        return 0.0
    labels = np.asarray([0] * len(normal_scores) + [1] * len(anomaly_scores), dtype=np.int32)
    scores = np.asarray(normal_scores + anomaly_scores, dtype=np.float32)
    return float(roc_auc_score(labels, scores) * 100.0)


def derive_manifest_name(manifest_repr: str) -> str:
    if manifest_repr.startswith("in_memory:"):
        return manifest_repr
    return Path(manifest_repr).name


def derive_shift_family(manifest_name: str, augmentation_types: list[str]) -> str:
    if manifest_name.startswith("query_") and manifest_name.endswith(".jsonl"):
        return manifest_name[len("query_") : -len(".jsonl")]
    if len(augmentation_types) == 1:
        return augmentation_types[0]
    return "multi"


def derive_severity_spec(entries: list[dict]) -> dict[str, object]:
    if not entries:
        return {}
    first_params = dict(entries[0].get("params", {}))
    for entry in entries[1:]:
        if dict(entry.get("params", {})) != first_params:
            return {"mixed": True}
    return first_params


def flatten_severity_spec(spec: dict[str, object]) -> dict[str, object]:
    return {f"severity_param_{key}": value for key, value in sorted(spec.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--augmentation-type")
    parser.add_argument("--augmentation-types", nargs="+")
    parser.add_argument("--severities", nargs="+")
    parser.add_argument("--input-root", default="data/query_normal_clean")
    parser.add_argument("--data-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/univad_manifest_shift",
    )
    parser.add_argument(
        "--log-dir",
        default="experiments/validation/condition_shift_baseline/reports/univad_manifest_shift/logs",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="regram-condition-shift")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="univad-manifest-shift")
    parser.add_argument(
        "--wandb-mode", default="online", choices=["online", "offline", "disabled"]
    )
    parser.add_argument("--wandb-log-images", action="store_true")
    parser.add_argument("--wandb-max-images", type=int, default=2)
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        raise RuntimeError(
            "UniVAD manifest-shift requires CUDA. Re-run on a GPU runtime with --device cuda."
        )

    phase_logs: list[str] = []
    run_started_at = time.perf_counter()
    log_phase(
        phase_logs,
        "run",
        f"start | category={args.category} | device={args.device} | manifest={args.manifest or 'in_memory'}",
    )

    phase_started_at = time.perf_counter()
    univad_root = ensure_external_on_path("univad")
    from UniVAD import UniVAD  # noqa: WPS433 — must come after path registration
    finish_phase(phase_logs, "imports", phase_started_at)

    device = torch.device(args.device)
    output_dir = REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_started_at = time.perf_counter()
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        all_entries = [
            entry
            for entry in load_manifest(manifest_path)
            if entry["category"] == args.category
        ]
        manifest_repr = str(manifest_path)
    elif args.augmentation_type or args.augmentation_types:
        augmentation_types = []
        if args.augmentation_type:
            augmentation_types.append(args.augmentation_type)
        if args.augmentation_types:
            augmentation_types.extend(args.augmentation_types)
        augmentation_types = list(dict.fromkeys(augmentation_types))
        all_entries = [
            entry
            for entry in build_manifest_entries(
                REPO_ROOT / args.input_root,
                augmentations=augmentation_types,
                severities=["low", "medium", "high"],
                seed=20260420,
            )
            if entry["category"] == args.category
        ]
        manifest_repr = f"in_memory:{','.join(augmentation_types)}"
    else:
        raise ValueError("Either --manifest or --augmentation-type(s) is required.")

    selected_severities = list(dict.fromkeys(args.severities or []))
    if selected_severities:
        all_entries = [
            entry for entry in all_entries if entry["severity"] in selected_severities
        ]
        if not all_entries:
            raise ValueError(
                f"No manifest entries matched category={args.category} and severities={selected_severities}."
            )

    grouped: dict[str, dict[str, list[dict]]] = {}
    augmentation_types_seen: list[str] = []
    for entry in all_entries:
        aug_type = entry["augmentation_type"]
        if aug_type not in augmentation_types_seen:
            augmentation_types_seen.append(aug_type)
        grouped.setdefault(aug_type, {}).setdefault(entry["severity"], []).append(entry)

    manifest_name = derive_manifest_name(manifest_repr)
    shift_family = derive_shift_family(manifest_name, augmentation_types_seen)
    severity_label = (
        selected_severities[0]
        if len(selected_severities) == 1
        else "multi"
        if selected_severities
        else "all"
    )
    severity_spec = derive_severity_spec(all_entries)
    severity_spec_flat = flatten_severity_spec(severity_spec)
    finish_phase(
        phase_logs,
        "manifest_prepare",
        phase_started_at,
        extra=(
            f"entries={len(all_entries)} | shift_family={shift_family} | "
            f"severities={','.join(selected_severities or ['low', 'medium', 'high'])} | "
            f"severity_spec={json.dumps(severity_spec, ensure_ascii=True, sort_keys=True)}"
        ),
    )

    transform = build_transform(args.image_size)
    data_root = REPO_ROOT / args.data_root
    train_good_dir = data_root / args.category / "train" / "good"
    clean_good_dir = data_root / args.category / "test" / "good"
    logical_dir = data_root / args.category / "test" / "logical_anomalies"
    structural_dir = data_root / args.category / "test" / "structural_anomalies"

    results: dict[str, object] = {}
    preview_images: dict[str, list[dict[str, object]]] = {}

    with univad_runtime_context(univad_root):
        phase_started_at = time.perf_counter()
        model = UniVAD(image_size=args.image_size, device=device).to(device)
        finish_phase(phase_logs, "model_build", phase_started_at)

        phase_started_at = time.perf_counter()
        reference_tensor, reference_paths = load_reference_tensors(
            train_good_dir,
            k_shot=args.k_shot,
            round_idx=args.round,
            transform=transform,
            device=device,
        )
        model.setup(
            {
                "few_shot_samples": reference_tensor,
                "dataset_category": args.category,
                "image_path": reference_paths,
            }
        )
        finish_phase(
            phase_logs,
            "setup_few_shot",
            phase_started_at,
            extra=f"k_shot={args.k_shot} | round={args.round}",
        )

        phase_started_at = time.perf_counter()
        clean_paths = sorted(clean_good_dir.glob("*.png"))
        clean_scores: list[float] = []
        for path in clean_paths:
            with Image.open(path) as image_obj:
                image = image_obj.convert("RGB")
            clean_scores.append(score_image(model, image, str(path), transform, device))
        finish_phase(
            phase_logs,
            "score_clean_good",
            phase_started_at,
            extra=f"samples={len(clean_paths)}",
        )

        phase_started_at = time.perf_counter()
        anomaly_paths = sorted(logical_dir.glob("*.png")) + sorted(
            structural_dir.glob("*.png")
        )
        anomaly_scores: list[float] = []
        for path in anomaly_paths:
            with Image.open(path) as image_obj:
                image = image_obj.convert("RGB")
            anomaly_scores.append(score_image(model, image, str(path), transform, device))
        finish_phase(
            phase_logs,
            "score_clean_anomaly",
            phase_started_at,
            extra=f"samples={len(anomaly_paths)}",
        )

        clean_threshold = float(np.max(clean_scores)) if clean_scores else 0.0
        results = {
            "updated_at": now_kst_string(),
            "category": args.category,
            "manifest": manifest_repr,
            "manifest_name": manifest_name,
            "shift_family": shift_family,
            "selected_severities": selected_severities or ["low", "medium", "high"],
            "severity_label": severity_label,
            "severity_spec": severity_spec,
            "augmentation_types": augmentation_types_seen,
            "threshold_policy": "clean_max",
            "clean_good": summarize_scores(clean_scores, clean_threshold),
            "clean_anomaly": summarize_scores(anomaly_scores, clean_threshold),
            "clean_image_auroc": compute_image_auroc(clean_scores, anomaly_scores),
            "augmentations": {},
        }

        for aug_type, severity_groups in sorted(grouped.items()):
            results["augmentations"][aug_type] = {}
            for severity, entries in sorted(severity_groups.items()):
                phase_started_at = time.perf_counter()
                shifted_scores: list[float] = []
                for entry in entries:
                    image_path = resolve_manifest_image_path(entry)
                    with Image.open(image_path) as image_obj:
                        image = image_obj.convert("RGB")
                    image = apply_augmentation(
                        image,
                        augmentation_type=entry["augmentation_type"],
                        severity=entry["severity"],
                        seed=entry["seed"],
                        params=entry["params"],
                    )
                    shifted_scores.append(
                        score_image(model, image, str(image_path), transform, device)
                    )
                summary = summarize_scores(shifted_scores, clean_threshold)
                summary["mean_score_shift"] = (
                    summary["mean"] - results["clean_good"]["mean"]
                )
                summary["image_auroc_vs_clean_anomaly"] = compute_image_auroc(
                    shifted_scores, anomaly_scores
                )
                results["augmentations"][aug_type][severity] = summary
                if args.wandb_log_images and args.wandb_max_images > 0:
                    preview_images[f"{aug_type}_{severity}"] = build_preview_images(
                        entries,
                        max_images=args.wandb_max_images,
                    )
                finish_phase(
                    phase_logs,
                    f"score_shifted::{aug_type}/{severity}",
                    phase_started_at,
                    extra=f"samples={len(entries)}",
                )

    output_suffix = f"{shift_family}_{severity_label}"
    output_path = output_dir / f"{args.category}_{output_suffix}.json"
    log_path = REPO_ROOT / args.log_dir / f"{args.category}_{output_suffix}.log.txt"
    run_name = f"univad-{args.category}-{output_suffix}-{datetime.now().strftime('%Y%m%d')}"
    wandb_tags = [
        "UniVAD",
        "mvtec_loco",
        "manifest_shift",
        f"shift:{shift_family}",
        f"severity:{severity_label}",
        f"class:{args.category}",
    ]
    if selected_severities:
        wandb_tags.extend([f"selected:{severity}" for severity in selected_severities])

    phase_started_at = time.perf_counter()
    wandb_run = init_wandb_run(
        enabled=args.use_wandb and args.wandb_mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        tags=wandb_tags,
        config={
            "baseline": "UniVAD",
            "dataset": "mvtec_loco",
            "class_name": args.category,
            "eval_type": "manifest_shift",
            "manifest": manifest_repr,
            "manifest_name": manifest_name,
            "shift_family": shift_family,
            "selected_severities": selected_severities or ["low", "medium", "high"],
            "severity": selected_severities[0] if len(selected_severities) == 1 else None,
            "severity_label": severity_label,
            "severity_spec": severity_spec,
            "augmentation_types": augmentation_types_seen,
            "input_root": str((REPO_ROOT / args.input_root).resolve()),
            "data_root": str(data_root.resolve()),
            "image_size": args.image_size,
            "k_shot": args.k_shot,
            "round": args.round,
            "device": args.device,
            "threshold_policy": "clean_max",
            "wandb_log_images": args.wandb_log_images,
            "wandb_max_images": args.wandb_max_images,
            "summary_path": str(output_path),
            "log_path": str(log_path),
            **severity_spec_flat,
        },
        mode=args.wandb_mode,
    )
    finish_phase(
        phase_logs,
        "wandb_init",
        phase_started_at,
        extra=f"enabled={bool(wandb_run is not None)}",
    )

    phase_started_at = time.perf_counter()
    summary = build_summary(
        baseline="UniVAD",
        dataset="mvtec_loco",
        class_name=args.category,
        eval_type="manifest_shift",
        device=str(device),
        output_path=output_path,
        log_path=log_path,
        config={
            "manifest": manifest_repr,
            "manifest_name": manifest_name,
            "shift_family": shift_family,
            "selected_severities": selected_severities or ["low", "medium", "high"],
            "severity": selected_severities[0] if len(selected_severities) == 1 else None,
            "severity_label": severity_label,
            "severity_spec": severity_spec,
            "augmentation_types": augmentation_types_seen,
            "input_root": str((REPO_ROOT / args.input_root).resolve()),
            "data_root": str(data_root.resolve()),
            "image_size": args.image_size,
            "k_shot": args.k_shot,
            "round": args.round,
            "device": args.device,
            "threshold_policy": "clean_max",
            "wandb_log_images": args.wandb_log_images,
            "wandb_max_images": args.wandb_max_images,
            **severity_spec_flat,
        },
        metrics={
            "clean_image_auroc": results["clean_image_auroc"],
            "clean_good_mean": results["clean_good"]["mean"],
            "clean_good_fpr_over_clean_max": results["clean_good"]["fpr_over_clean_max"],
            "clean_anomaly_mean": results["clean_anomaly"]["mean"],
            "clean_anomaly_fpr_over_clean_max": results["clean_anomaly"][
                "fpr_over_clean_max"
            ],
        },
        paths={
            "repo_root": REPO_ROOT,
            "input_root": (REPO_ROOT / args.input_root).resolve(),
            "data_root": data_root.resolve(),
            "univad_root": univad_root,
        },
        payload=results,
    )
    finish_phase(phase_logs, "build_summary", phase_started_at)

    phase_started_at = time.perf_counter()
    write_summary(summary, output_path)
    finish_phase(
        phase_logs,
        "write_outputs",
        phase_started_at,
        extra=f"summary={output_path.name}",
    )

    phase_started_at = time.perf_counter()
    log_summary_to_wandb(
        wandb_run,
        summary=summary,
        summary_path=output_path,
        log_path=log_path,
    )
    log_preview_images_to_wandb(wandb_run, preview_images=preview_images)
    finish_wandb_run(wandb_run)
    finish_phase(phase_logs, "wandb_finalize", phase_started_at)
    finish_phase(phase_logs, "run", run_started_at, extra=f"output={output_path.name}")
    write_log(
        log_path,
        [
            "runner=run_univad_manifest_shift.py",
            "baseline=UniVAD",
            "dataset=mvtec_loco",
            f"class_name={args.category}",
            "eval_type=manifest_shift",
            f"manifest={manifest_repr}",
            f"manifest_name={manifest_name}",
            f"shift_family={shift_family}",
            f"selected_severities={','.join(selected_severities or ['low', 'medium', 'high'])}",
            f"severity_spec={json.dumps(severity_spec, ensure_ascii=True, sort_keys=True)}",
            f"augmentation_types={','.join(augmentation_types_seen)}",
            f"output_path={output_path}",
            *phase_logs,
        ],
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
