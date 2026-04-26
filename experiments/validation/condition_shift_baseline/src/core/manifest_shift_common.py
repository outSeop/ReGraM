"""Shared helpers for manifest-based condition-shift runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from augmentation_runtime import build_manifest_entries, load_manifest
from contracts import build_summary, write_summary
from repo_paths import REPO_ROOT
from wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_preview_images_to_wandb,
    log_summary_to_wandb,
)


DEFAULT_SELECTED_SEVERITIES = ("low", "medium", "high")
THRESHOLD_POLICY_CLEAN_MAX = "clean_max"
DEFAULT_GALLERY_SAMPLE_LIMIT = 6


@dataclass(frozen=True)
class ManifestShiftRunSpec:
    category: str
    manifest_repr: str
    manifest_name: str
    shift_family: str
    grouped_entries: dict[str, dict[str, list[dict[str, Any]]]]
    augmentation_types_seen: list[str]
    selected_severities: list[str]
    severity_label: str
    severity_spec: dict[str, object]
    severity_spec_flat: dict[str, object]
    output_suffix: str
    total_entries: int

    @property
    def effective_selected_severities(self) -> list[str]:
        return self.selected_severities or list(DEFAULT_SELECTED_SEVERITIES)


@dataclass(frozen=True)
class ManifestShiftOutputPaths:
    output_dir: Path
    output_path: Path
    log_path: Path


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _derive_manifest_name(manifest_repr: str) -> str:
    if manifest_repr.startswith("in_memory:") or manifest_repr.startswith("multi:"):
        return manifest_repr
    return Path(manifest_repr).name


def _derive_shift_family(manifest_name: str, augmentation_types: list[str]) -> str:
    if manifest_name.startswith("query_") and manifest_name.endswith(".jsonl"):
        return manifest_name[len("query_") : -len(".jsonl")]
    if len(augmentation_types) == 1:
        return augmentation_types[0]
    return "multi"


def _derive_severity_spec(entries: list[dict[str, Any]]) -> dict[str, object]:
    if not entries:
        return {}
    first_params = dict(entries[0].get("params", {}))
    for entry in entries[1:]:
        if dict(entry.get("params", {})) != first_params:
            return {"mixed": True}
    return first_params


def _flatten_severity_spec(spec: dict[str, object]) -> dict[str, object]:
    return {f"severity_param_{key}": value for key, value in sorted(spec.items())}


def prepare_manifest_shift_run_spec(
    *,
    category: str,
    input_root: str | Path,
    manifest_paths: list[str] | None = None,
    augmentation_type: str | None = None,
    augmentation_types: list[str] | None = None,
    severities: list[str] | None = None,
) -> ManifestShiftRunSpec:
    manifest_paths_cli = list(manifest_paths or [])

    if manifest_paths_cli:
        resolved_manifest_paths = [resolve_repo_path(raw_path) for raw_path in manifest_paths_cli]
        all_entries = [
            entry
            for manifest_path in resolved_manifest_paths
            for entry in load_manifest(manifest_path)
            if entry["category"] == category
        ]
        if len(resolved_manifest_paths) == 1:
            manifest_repr = str(resolved_manifest_paths[0])
        else:
            manifest_repr = "multi:" + ",".join(path.name for path in resolved_manifest_paths)
    elif augmentation_type or augmentation_types:
        deduped_augmentation_types: list[str] = []
        if augmentation_type:
            deduped_augmentation_types.append(augmentation_type)
        if augmentation_types:
            deduped_augmentation_types.extend(augmentation_types)
        deduped_augmentation_types = list(dict.fromkeys(deduped_augmentation_types))
        all_entries = [
            entry
            for entry in build_manifest_entries(
                resolve_repo_path(input_root),
                augmentations=deduped_augmentation_types,
                severities=list(DEFAULT_SELECTED_SEVERITIES),
                seed=20260420,
            )
            if entry["category"] == category
        ]
        manifest_repr = f"in_memory:{','.join(deduped_augmentation_types)}"
    else:
        raise ValueError("Either --manifest or --augmentation-type(s) is required.")

    selected_severities = list(dict.fromkeys(severities or []))
    if selected_severities:
        all_entries = [entry for entry in all_entries if entry["severity"] in selected_severities]
        if not all_entries:
            raise ValueError(
                f"No manifest entries matched category={category} and severities={selected_severities}."
            )

    grouped_entries: dict[str, dict[str, list[dict[str, Any]]]] = {}
    augmentation_types_seen: list[str] = []
    for entry in all_entries:
        aug_type = entry["augmentation_type"]
        if aug_type not in augmentation_types_seen:
            augmentation_types_seen.append(aug_type)
        grouped_entries.setdefault(aug_type, {}).setdefault(entry["severity"], []).append(entry)

    manifest_name = _derive_manifest_name(manifest_repr)
    shift_family = _derive_shift_family(manifest_name, augmentation_types_seen)
    severity_label = (
        selected_severities[0]
        if len(selected_severities) == 1
        else "multi"
        if selected_severities
        else "all"
    )
    severity_spec = _derive_severity_spec(all_entries)
    return ManifestShiftRunSpec(
        category=category,
        manifest_repr=manifest_repr,
        manifest_name=manifest_name,
        shift_family=shift_family,
        grouped_entries=grouped_entries,
        augmentation_types_seen=augmentation_types_seen,
        selected_severities=selected_severities,
        severity_label=severity_label,
        severity_spec=severity_spec,
        severity_spec_flat=_flatten_severity_spec(severity_spec),
        output_suffix=f"{shift_family}_{severity_label}",
        total_entries=len(all_entries),
    )


def build_manifest_prepare_extra(run_spec: ManifestShiftRunSpec) -> str:
    return (
        f"entries={run_spec.total_entries} | shift_family={run_spec.shift_family} | "
        f"severities={','.join(run_spec.effective_selected_severities)} | "
        f"severity_spec={json.dumps(run_spec.severity_spec, ensure_ascii=True, sort_keys=True)}"
    )


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _percentile(array: np.ndarray, q: float) -> float:
    return float(np.percentile(array, q)) if array.size else 0.0


def summarize_scores(scores: list[float], threshold: float) -> dict[str, float | int]:
    array = np.asarray(scores, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "median": _percentile(array, 50.0),
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "p25": _percentile(array, 25.0),
        "p75": _percentile(array, 75.0),
        "p95": _percentile(array, 95.0),
        "max": float(array.max()) if array.size else 0.0,
        "fpr_over_clean_max": float((array > threshold).mean()) if array.size else 0.0,
    }


def build_results_scaffold(
    *,
    updated_at: str,
    category: str,
    run_spec: ManifestShiftRunSpec,
    clean_good: dict[str, Any],
    clean_anomaly: dict[str, Any],
    clean_image_auroc: float,
) -> dict[str, Any]:
    return {
        "updated_at": updated_at,
        "category": category,
        "manifest": run_spec.manifest_repr,
        "manifest_name": run_spec.manifest_name,
        "shift_family": run_spec.shift_family,
        "selected_severities": run_spec.effective_selected_severities,
        "severity_label": run_spec.severity_label,
        "severity_spec": run_spec.severity_spec,
        "severity_spec_by_cell": {},
        "augmentation_types": run_spec.augmentation_types_seen,
        "threshold_policy": THRESHOLD_POLICY_CLEAN_MAX,
        "clean_good": clean_good,
        "clean_anomaly": clean_anomaly,
        "clean_image_auroc": clean_image_auroc,
        "augmentations": {},
    }


def record_shift_cell(
    results: dict[str, Any],
    *,
    aug_type: str,
    severity: str,
    summary: dict[str, Any],
    entries: list[dict[str, Any]],
) -> None:
    results["augmentations"].setdefault(aug_type, {})[severity] = summary
    results["severity_spec_by_cell"][f"{aug_type}/{severity}"] = dict(entries[0].get("params", {}))


def build_shift_sample_rows(
    *,
    entries: list[dict[str, Any]],
    scores: list[float],
    clean_good_mean: float,
    max_items: int = DEFAULT_GALLERY_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    from manifest_paths import resolve_manifest_image_path  # noqa: WPS433

    rows: list[dict[str, Any]] = []
    for entry, score in list(zip(entries, scores, strict=False))[:max_items]:
        image_path = resolve_manifest_image_path(entry)
        rows.append(
            {
                "source_id": entry.get("source_id"),
                "image_path": str(image_path),
                "image_name": Path(image_path).name,
                "score": float(score),
                "score_delta_from_clean_mean": float(score - clean_good_mean),
                "augmentation_type": entry.get("augmentation_type"),
                "severity": entry.get("severity"),
                "seed": entry.get("seed"),
                "params": dict(entry.get("params", {})),
            }
        )
    return rows


def prepare_output_paths(
    *,
    output_root: str | Path,
    log_dir: str | Path,
    category: str,
    output_suffix: str,
) -> ManifestShiftOutputPaths:
    output_dir = resolve_repo_path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir_path = resolve_repo_path(log_dir)
    return ManifestShiftOutputPaths(
        output_dir=output_dir,
        output_path=output_dir / f"{category}_{output_suffix}.json",
        log_path=log_dir_path / f"{category}_{output_suffix}.log.txt",
    )


def build_run_name(baseline_slug: str, category: str, output_suffix: str) -> str:
    return f"{baseline_slug}-{category}-{output_suffix}-{datetime.now().strftime('%Y%m%d')}"


def build_common_run_config(
    *,
    run_spec: ManifestShiftRunSpec,
    input_root: str | Path,
    data_root: str | Path,
    device: str,
    wandb_log_images: bool,
    wandb_max_images: int,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "manifest": run_spec.manifest_repr,
        "manifest_name": run_spec.manifest_name,
        "shift_family": run_spec.shift_family,
        "selected_severities": run_spec.effective_selected_severities,
        "severity": run_spec.selected_severities[0] if len(run_spec.selected_severities) == 1 else None,
        "severity_label": run_spec.severity_label,
        "severity_spec": run_spec.severity_spec,
        "augmentation_types": run_spec.augmentation_types_seen,
        "input_root": str(resolve_repo_path(input_root).resolve()),
        "data_root": str(resolve_repo_path(data_root).resolve()),
        "device": device,
        "threshold_policy": THRESHOLD_POLICY_CLEAN_MAX,
        "wandb_log_images": wandb_log_images,
        "wandb_max_images": wandb_max_images,
    }
    if extra_config:
        config.update(extra_config)
    config.update(run_spec.severity_spec_flat)
    return config


def build_clean_metric_snapshot(results: dict[str, Any]) -> dict[str, Any]:
    clean_good = results["clean_good"]
    clean_anomaly = results["clean_anomaly"]
    return {
        "clean_image_auroc": results["clean_image_auroc"],
        "clean_good_mean": clean_good["mean"],
        "clean_good_median": clean_good["median"],
        "clean_good_fpr_over_clean_max": clean_good["fpr_over_clean_max"],
        "clean_anomaly_mean": clean_anomaly["mean"],
        "clean_anomaly_median": clean_anomaly["median"],
        "clean_anomaly_fpr_over_clean_max": clean_anomaly["fpr_over_clean_max"],
    }


def build_shift_metric_snapshot(results: dict[str, Any]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for aug_type, severity_map in sorted(results.get("augmentations", {}).items()):
        for severity, item in sorted(severity_map.items()):
            cells.append(
                {
                    "cell_key": f"{aug_type}/{severity}",
                    "severity": severity,
                    "shifted_normal_fpr": item.get(
                        "shifted_normal_fpr",
                        item.get("fpr_over_clean_max"),
                    ),
                    "image_auroc_drop_from_clean": item.get("image_auroc_drop_from_clean"),
                    "mean_score_shift": item.get("mean_score_shift"),
                    "median_score_shift": item.get("median_score_shift"),
                }
            )

    if not cells:
        return {}

    fpr_values = [
        float(cell["shifted_normal_fpr"])
        for cell in cells
        if cell["shifted_normal_fpr"] is not None
    ]
    auroc_drop_values = [
        float(cell["image_auroc_drop_from_clean"])
        for cell in cells
        if cell["image_auroc_drop_from_clean"] is not None
    ]
    mean_shift_values = [
        float(cell["mean_score_shift"])
        for cell in cells
        if cell["mean_score_shift"] is not None
    ]
    median_shift_values = [
        float(cell["median_score_shift"])
        for cell in cells
        if cell["median_score_shift"] is not None
    ]

    worst_fpr_cell = max(
        cells,
        key=lambda cell: cell["shifted_normal_fpr"]
        if cell["shifted_normal_fpr"] is not None
        else -1.0,
    )
    worst_auroc_drop_cell = max(
        cells,
        key=lambda cell: cell["image_auroc_drop_from_clean"]
        if cell["image_auroc_drop_from_clean"] is not None
        else -1.0,
    )

    snapshot: dict[str, Any] = {
        "mean_shifted_normal_fpr": _mean_or_none(fpr_values),
        "worst_shifted_normal_fpr": _max_or_none(fpr_values),
        "mean_fpr_over_clean_max": _mean_or_none(fpr_values),
        "worst_fpr_over_clean_max": _max_or_none(fpr_values),
        "worst_shift_cell_by_fpr": worst_fpr_cell["cell_key"],
        "worst_cell": worst_fpr_cell["cell_key"],
        "mean_image_auroc_drop_from_clean": _mean_or_none(auroc_drop_values),
        "worst_image_auroc_drop_from_clean": _max_or_none(auroc_drop_values),
        "worst_shift_cell_by_auroc_drop": worst_auroc_drop_cell["cell_key"],
        "mean_score_shift": _mean_or_none(mean_shift_values),
        "mean_median_score_shift": _mean_or_none(median_shift_values),
    }

    severity_buckets: dict[str, dict[str, list[float]]] = {}
    for cell in cells:
        bucket = severity_buckets.setdefault(
            cell["severity"],
            {
                "shifted_normal_fpr": [],
                "image_auroc_drop_from_clean": [],
                "mean_score_shift": [],
                "median_score_shift": [],
            },
        )
        for key in bucket:
            value = cell.get(key)
            if value is not None:
                bucket[key].append(float(value))

    for severity, bucket in sorted(severity_buckets.items()):
        snapshot[f"mean_shifted_normal_fpr_by_severity/{severity}"] = _mean_or_none(
            bucket["shifted_normal_fpr"]
        )
        snapshot[f"mean_fpr_by_severity/{severity}"] = _mean_or_none(
            bucket["shifted_normal_fpr"]
        )
        snapshot[f"mean_image_auroc_drop_from_clean_by_severity/{severity}"] = _mean_or_none(
            bucket["image_auroc_drop_from_clean"]
        )
        snapshot[f"mean_score_shift_by_severity/{severity}"] = _mean_or_none(
            bucket["mean_score_shift"]
        )
        snapshot[f"mean_median_score_shift_by_severity/{severity}"] = _mean_or_none(
            bucket["median_score_shift"]
        )

    return snapshot


def _build_wandb_tags(
    *,
    baseline: str,
    class_name: str,
    run_spec: ManifestShiftRunSpec,
) -> list[str]:
    tags = [
        baseline,
        "mvtec_loco",
        "manifest_shift",
        f"shift:{run_spec.shift_family}",
        f"severity:{run_spec.severity_label}",
        f"class:{class_name}",
    ]
    if run_spec.selected_severities:
        tags.extend([f"selected:{severity}" for severity in run_spec.selected_severities])
    return tags


def init_manifest_shift_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    group: str,
    mode: str,
    baseline: str,
    dataset: str,
    class_name: str,
    eval_type: str,
    run_name: str,
    config: dict[str, Any],
    run_spec: ManifestShiftRunSpec,
):
    return init_wandb_run(
        enabled=enabled,
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        tags=_build_wandb_tags(
            baseline=baseline,
            class_name=class_name,
            run_spec=run_spec,
        ),
        config={
            "baseline": baseline,
            "dataset": dataset,
            "class_name": class_name,
            "eval_type": eval_type,
            **config,
        },
        mode=mode,
    )


def build_manifest_shift_summary(
    *,
    baseline: str,
    dataset: str,
    class_name: str,
    eval_type: str,
    device: str,
    output_paths: ManifestShiftOutputPaths,
    config: dict[str, Any],
    metrics: dict[str, Any],
    paths: dict[str, Any],
    payload: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_summary(
        baseline=baseline,
        dataset=dataset,
        class_name=class_name,
        eval_type=eval_type,
        device=device,
        output_path=output_paths.output_path,
        log_path=output_paths.log_path,
        config=config,
        metrics=metrics,
        paths=paths,
        payload=payload,
        artifacts=artifacts,
    )


def write_manifest_shift_summary(
    summary: dict[str, Any],
    *,
    output_paths: ManifestShiftOutputPaths,
) -> None:
    write_summary(summary, output_paths.output_path)


def finalize_manifest_shift_tracking(
    run,
    *,
    summary: dict[str, Any],
    output_paths: ManifestShiftOutputPaths,
    preview_images: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    log_summary_to_wandb(
        run,
        summary=summary,
        summary_path=output_paths.output_path,
        log_path=output_paths.log_path,
    )
    log_preview_images_to_wandb(run, preview_images=preview_images or {})
    finish_wandb_run(run)


def build_manifest_shift_log_lines(
    *,
    run_spec: ManifestShiftRunSpec,
    output_path: Path,
) -> list[str]:
    return [
        f"manifest={run_spec.manifest_repr}",
        f"manifest_name={run_spec.manifest_name}",
        f"shift_family={run_spec.shift_family}",
        f"selected_severities={','.join(run_spec.effective_selected_severities)}",
        f"severity_spec={json.dumps(run_spec.severity_spec, ensure_ascii=True, sort_keys=True)}",
        f"augmentation_types={','.join(run_spec.augmentation_types_seen)}",
        f"output_path={output_path}",
    ]
