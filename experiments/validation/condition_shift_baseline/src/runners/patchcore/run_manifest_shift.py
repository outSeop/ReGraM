"""Run PatchCore condition-shift evaluation from a manifest."""

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
import os
import time

import numpy as np
import torch

from manifest_shift_common import (
    attach_heatmap_artifacts,
    build_clean_metric_snapshot,
    build_common_run_config,
    build_clean_sample_rows,
    build_manifest_prepare_extra,
    build_manifest_shift_log_lines,
    build_manifest_shift_summary,
    build_results_scaffold,
    build_run_name,
    build_shift_sample_rows,
    build_shift_metric_snapshot,
    finalize_manifest_shift_tracking,
    init_manifest_shift_wandb_run,
    prepare_manifest_shift_run_spec,
    prepare_output_paths,
    record_shift_cell,
    resolve_repo_path,
    summarize_scores,
    write_manifest_shift_summary,
)
from patchcore_datasets import ImageFolderDataset, ManifestSubsetDataset, MultiFolderDataset
from patchcore_factory import build_patchcore, get_patchcore, make_loader
from preview_utils import build_preview_images
from repo_paths import REPO_ROOT, finish_phase, log_phase, now_kst_string
from contracts import write_log


def compute_image_auroc(normal_scores: list[float], anomaly_scores: list[float]) -> float:
    ns = get_patchcore()
    labels = np.asarray([0] * len(normal_scores) + [1] * len(anomaly_scores), dtype=np.int32)
    scores = np.asarray(normal_scores + anomaly_scores, dtype=np.float32)
    return float(ns.metrics.compute_imagewise_retrieval_metrics(scores, labels)["auroc"] * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--manifest", nargs="+")
    parser.add_argument("--augmentation-type")
    parser.add_argument("--augmentation-types", nargs="+")
    parser.add_argument("--severities", nargs="+")
    parser.add_argument("--input-root", default="data/query_normal_clean")
    parser.add_argument("--data-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampler-percentage", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/patchcore_manifest_shift",
    )
    parser.add_argument(
        "--log-dir",
        default="experiments/validation/condition_shift_baseline/reports/patchcore_manifest_shift/logs",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="regram-condition-shift")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="patchcore")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-images", action="store_true")
    parser.add_argument("--wandb-max-images", type=int, default=2)
    parser.add_argument("--export-heatmaps", action="store_true")
    parser.add_argument("--heatmap-max-images", type=int, default=2)
    args = parser.parse_args()

    manifest_paths_cli = list(args.manifest or [])
    manifest_start_label = ",".join(manifest_paths_cli) if manifest_paths_cli else "in_memory"

    phase_logs: list[str] = []
    run_started_at = time.perf_counter()
    log_phase(
        phase_logs,
        "run",
        f"start | category={args.category} | device={args.device} | manifest={manifest_start_label}",
    )

    phase_started_at = time.perf_counter()
    ns = get_patchcore()
    finish_phase(phase_logs, "imports", phase_started_at)

    device = torch.device(args.device)
    data_root = resolve_repo_path(args.data_root)
    input_root = resolve_repo_path(args.input_root)

    phase_started_at = time.perf_counter()
    train_dataset = ns.MVTecDataset(
        str(data_root),
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=ns.DatasetSplit.TRAIN,
    )
    clean_dataset = ImageFolderDataset(
        data_root / args.category / "test" / "good",
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )
    logical_anomaly_dataset = MultiFolderDataset(
        [data_root / args.category / "test" / "logical_anomalies"],
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )
    structural_anomaly_dataset = MultiFolderDataset(
        [data_root / args.category / "test" / "structural_anomalies"],
        category=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
    )
    anomaly_count = len(logical_anomaly_dataset) + len(structural_anomaly_dataset)
    finish_phase(
        phase_logs,
        "dataset_setup",
        phase_started_at,
        extra=(
            f"train={len(train_dataset)} | clean={len(clean_dataset)} | "
            f"logical={len(logical_anomaly_dataset)} | structural={len(structural_anomaly_dataset)} | "
            f"anomaly={anomaly_count}"
        ),
    )

    phase_started_at = time.perf_counter()
    run_spec = prepare_manifest_shift_run_spec(
        category=args.category,
        input_root=args.input_root,
        manifest_paths=manifest_paths_cli,
        augmentation_type=args.augmentation_type,
        augmentation_types=args.augmentation_types,
        severities=args.severities,
    )
    finish_phase(
        phase_logs,
        "manifest_prepare",
        phase_started_at,
        extra=build_manifest_prepare_extra(run_spec),
    )

    output_paths = prepare_output_paths(
        output_root=args.output,
        log_dir=args.log_dir,
        category=args.category,
        output_suffix=run_spec.output_suffix,
    )
    explainability_dir = output_paths.output_dir / "explainability" / args.category / run_spec.output_suffix

    phase_started_at = time.perf_counter()
    model = build_patchcore(device, args.imagesize, args.num_workers, args.sampler_percentage)
    finish_phase(phase_logs, "model_build", phase_started_at)

    phase_started_at = time.perf_counter()
    model.fit(make_loader(train_dataset, args.batch_size, args.num_workers))
    finish_phase(phase_logs, "fit_train_good", phase_started_at, extra=f"samples={len(train_dataset)}")

    phase_started_at = time.perf_counter()
    clean_scores, clean_segmentations, _, _ = model.predict(make_loader(clean_dataset, args.batch_size, args.num_workers))
    clean_scores = [float(score) for score in clean_scores]
    finish_phase(phase_logs, "score_clean_good", phase_started_at, extra=f"samples={len(clean_dataset)}")

    phase_started_at = time.perf_counter()
    logical_anomaly_scores, logical_anomaly_segmentations, _, _ = model.predict(
        make_loader(logical_anomaly_dataset, args.batch_size, args.num_workers)
    )
    logical_anomaly_scores = [float(score) for score in logical_anomaly_scores]
    finish_phase(
        phase_logs,
        "score_clean_logical_anomaly",
        phase_started_at,
        extra=f"samples={len(logical_anomaly_dataset)}",
    )

    phase_started_at = time.perf_counter()
    structural_anomaly_scores, structural_anomaly_segmentations, _, _ = model.predict(
        make_loader(structural_anomaly_dataset, args.batch_size, args.num_workers)
    )
    structural_anomaly_scores = [float(score) for score in structural_anomaly_scores]
    finish_phase(
        phase_logs,
        "score_clean_structural_anomaly",
        phase_started_at,
        extra=f"samples={len(structural_anomaly_dataset)}",
    )
    anomaly_scores = logical_anomaly_scores + structural_anomaly_scores

    clean_threshold = max(clean_scores) if clean_scores else 0.0
    results = build_results_scaffold(
        updated_at=now_kst_string(),
        category=args.category,
        run_spec=run_spec,
        clean_good=summarize_scores(clean_scores, clean_threshold),
        clean_anomaly=summarize_scores(anomaly_scores, clean_threshold),
        clean_image_auroc=compute_image_auroc(clean_scores, anomaly_scores),
        clean_logical_anomaly=summarize_scores(logical_anomaly_scores, clean_threshold),
        clean_structural_anomaly=summarize_scores(structural_anomaly_scores, clean_threshold),
        clean_logical_image_auroc=compute_image_auroc(clean_scores, logical_anomaly_scores),
        clean_structural_image_auroc=compute_image_auroc(clean_scores, structural_anomaly_scores),
    )
    results["clean_score_distributions"] = {
        "clean_normal_scores": clean_scores,
        "clean_anomaly_scores": anomaly_scores,
        "clean_logical_anomaly_scores": logical_anomaly_scores,
        "clean_structural_anomaly_scores": structural_anomaly_scores,
    }
    if args.export_heatmaps:
        clean_reference_maps: dict[str, list[dict[str, object]]] = {}
        clean_reference_specs = [
            ("clean_normal", clean_dataset.image_paths, clean_scores, clean_segmentations),
            (
                "logical_anomaly",
                logical_anomaly_dataset.image_paths,
                logical_anomaly_scores,
                logical_anomaly_segmentations,
            ),
            (
                "structural_anomaly",
                structural_anomaly_dataset.image_paths,
                structural_anomaly_scores,
                structural_anomaly_segmentations,
            ),
        ]
        for sample_type, image_paths, scores_for_split, segmentations_for_split in clean_reference_specs:
            sample_rows = build_clean_sample_rows(
                image_paths=image_paths,
                scores=scores_for_split,
                clean_good_mean=results["clean_good"]["mean"],
                sample_type=sample_type,
                max_items=args.heatmap_max_images,
            )
            heatmaps_by_index = {
                int(row["source_index"]): segmentations_for_split[int(row["source_index"])]
                for row in sample_rows
            }
            clean_reference_maps[sample_type] = attach_heatmap_artifacts(
                sample_rows=sample_rows,
                heatmaps_by_index=heatmaps_by_index,
                artifact_dir=explainability_dir / "clean_reference" / sample_type,
                artifact_prefix=f"patchcore_{sample_type}",
                apply_shift=False,
            )
        results["clean_reference_maps"] = clean_reference_maps
    preview_images: dict[str, list[dict[str, object]]] = {}

    for aug_type, severity_groups in sorted(run_spec.grouped_entries.items()):
        for severity, entries in sorted(severity_groups.items()):
            phase_started_at = time.perf_counter()
            dataset = ManifestSubsetDataset(
                entries,
                category=args.category,
                resize=args.resize,
                imagesize=args.imagesize,
            )
            scores, segmentations, _, _ = model.predict(make_loader(dataset, args.batch_size, args.num_workers))
            scores = [float(score) for score in scores]
            summary = summarize_scores(scores, clean_threshold)
            summary["shifted_normal_fpr"] = summary["fpr_over_clean_max"]
            summary["mean_score_shift"] = summary["mean"] - results["clean_good"]["mean"]
            summary["median_score_shift"] = summary["median"] - results["clean_good"]["median"]
            summary["shifted_image_auroc"] = compute_image_auroc(scores, anomaly_scores)
            summary["image_auroc_vs_clean_anomaly"] = summary["shifted_image_auroc"]
            summary["shifted_image_auroc_vs_logical_anomaly"] = compute_image_auroc(
                scores,
                logical_anomaly_scores,
            )
            summary["shifted_image_auroc_vs_structural_anomaly"] = compute_image_auroc(
                scores,
                structural_anomaly_scores,
            )
            summary["image_auroc_drop_from_clean"] = (
                results["clean_image_auroc"] - summary["shifted_image_auroc"]
            )
            summary["image_auroc_drop_from_clean_logical"] = (
                results["clean_logical_image_auroc"]
                - summary["shifted_image_auroc_vs_logical_anomaly"]
            )
            summary["image_auroc_drop_from_clean_structural"] = (
                results["clean_structural_image_auroc"]
                - summary["shifted_image_auroc_vs_structural_anomaly"]
            )
            summary["shifted_normal_scores"] = scores
            summary["sample_rows"] = build_shift_sample_rows(
                entries=entries,
                scores=scores,
                clean_good_mean=results["clean_good"]["mean"],
                max_items=max(args.heatmap_max_images, args.wandb_max_images),
            )
            if args.export_heatmaps:
                heatmaps_by_index = {
                    int(row["source_index"]): segmentations[int(row["source_index"])]
                    for row in summary["sample_rows"]
                }
                summary["sample_rows"] = attach_heatmap_artifacts(
                    sample_rows=summary["sample_rows"],
                    heatmaps_by_index=heatmaps_by_index,
                    artifact_dir=explainability_dir / "shifted" / aug_type / severity,
                    artifact_prefix=f"patchcore_{aug_type}_{severity}",
                    apply_shift=True,
                )
            record_shift_cell(
                results,
                aug_type=aug_type,
                severity=severity,
                summary=summary,
                entries=entries,
            )
            if args.wandb_log_images and args.wandb_max_images > 0:
                preview_images[f"{aug_type}_{severity}"] = build_preview_images(
                    entries,
                    max_images=args.wandb_max_images,
                )
            finish_phase(
                phase_logs,
                f"score_shifted::{aug_type}/{severity}",
                phase_started_at,
                extra=f"samples={len(dataset)}",
            )

    common_run_config = build_common_run_config(
        run_spec=run_spec,
        input_root=args.input_root,
        data_root=args.data_root,
        device=args.device,
        wandb_log_images=args.wandb_log_images,
        wandb_max_images=args.wandb_max_images,
        extra_config={
            "resize": args.resize,
            "imagesize": args.imagesize,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "sampler_percentage": args.sampler_percentage,
            "export_heatmaps": args.export_heatmaps,
            "heatmap_max_images": args.heatmap_max_images,
        },
    )

    phase_started_at = time.perf_counter()
    wandb_error = "-"
    if args.use_wandb and args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("warning: WANDB_API_KEY is not set; W&B online init may be skipped.")
    try:
        wandb_run = init_manifest_shift_wandb_run(
            enabled=args.use_wandb and args.wandb_mode != "disabled",
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            mode=args.wandb_mode,
            baseline="PatchCore",
            dataset="mvtec_loco",
            class_name=args.category,
            eval_type="manifest_shift",
            run_name=build_run_name("patchcore", args.category, run_spec.output_suffix),
            config={
                **common_run_config,
                "summary_path": str(output_paths.output_path),
                "log_path": str(output_paths.log_path),
            },
            run_spec=run_spec,
        )
    except Exception as exc:  # noqa: BLE001
        wandb_run = None
        wandb_error = f"{type(exc).__name__}: {exc}"
        print(f"warning: W&B init failed; continuing without W&B. {wandb_error}")
    finish_phase(
        phase_logs,
        "wandb_init",
        phase_started_at,
        extra=f"enabled={bool(wandb_run is not None)} | error={wandb_error}",
    )

    phase_started_at = time.perf_counter()
    summary = build_manifest_shift_summary(
        baseline="PatchCore",
        dataset="mvtec_loco",
        class_name=args.category,
        eval_type="manifest_shift",
        device=str(device),
        output_paths=output_paths,
        config=common_run_config,
        metrics={
            **build_clean_metric_snapshot(results),
            **build_shift_metric_snapshot(results),
        },
        paths={
            "repo_root": REPO_ROOT,
            "input_root": input_root.resolve(),
            "data_root": data_root.resolve(),
        },
        payload=results,
    )
    finish_phase(phase_logs, "build_summary", phase_started_at)

    phase_started_at = time.perf_counter()
    write_manifest_shift_summary(summary, output_paths=output_paths)
    finish_phase(phase_logs, "write_outputs", phase_started_at, extra=f"summary={output_paths.output_path.name}")

    phase_started_at = time.perf_counter()
    try:
        finalize_manifest_shift_tracking(
            wandb_run,
            summary=summary,
            output_paths=output_paths,
            preview_images=preview_images,
        )
    except Exception as exc:  # noqa: BLE001
        wandb_error = f"{type(exc).__name__}: {exc}"
        print(f"warning: W&B finalize failed; outputs are still saved locally. {wandb_error}")
    finish_phase(phase_logs, "wandb_finalize", phase_started_at, extra=f"error={wandb_error}")

    finish_phase(phase_logs, "run", run_started_at, extra=f"output={output_paths.output_path.name}")
    write_log(
        output_paths.log_path,
        [
            "runner=run_patchcore_manifest_shift.py",
            "baseline=PatchCore",
            "dataset=mvtec_loco",
            f"class_name={args.category}",
            "eval_type=manifest_shift",
            *build_manifest_shift_log_lines(run_spec=run_spec, output_path=output_paths.output_path),
            *phase_logs,
        ],
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
