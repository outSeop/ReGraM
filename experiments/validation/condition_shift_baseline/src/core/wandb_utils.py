"""Minimal wandb helpers for experiment tracking.

Role:
- optional tracking layer
- keep wandb integration out of the main runner logic
- no-op unless the runner explicitly enables wandb
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from repo_paths import REPO_ROOT


SHIFT_CELL_COLUMNS = (
    "shift",
    "severity",
    "shifted_normal_mean",
    "shifted_normal_median",
    "p25",
    "p75",
    "shifted_normal_fpr",
    "mean_score_shift",
    "median_score_shift",
    "shifted_image_auroc",
    "image_auroc_drop_from_clean",
    "severity_param",
)

SHIFT_SUMMARY_TABLE_COLUMNS = (
    "baseline",
    "category",
    "corruption_type",
    "severity",
    "shifted_normal_mean",
    "shifted_normal_median",
    "p25",
    "p75",
    "shifted_normal_fpr",
    "shifted_image_auroc",
    "image_auroc_drop_from_clean",
    "mean_score_shift",
    "median_score_shift",
)


def _load_env_file(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE file without overriding existing vars.

    No deps, no interpolation, no multi-line values — matches the 12-factor
    subset of .env semantics that's enough for API keys. Existing env vars win.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    group: str,
    name: str,
    tags: list[str],
    config: dict[str, Any],
    mode: str,
):
    if not enabled:
        return None

    import wandb  # noqa: WPS433

    _load_env_file(REPO_ROOT / ".env")

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    elif mode == "online":
        raise RuntimeError(
            "WANDB_API_KEY is not set and --wandb-mode=online. "
            "Set it in the process env or add WANDB_API_KEY=... to "
            f"{REPO_ROOT / '.env'} (see .env.example). "
            "W&B logging will be skipped by runners that guard optional tracking."
        )

    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        tags=tags,
        config=config,
        mode=mode,
    )


def _build_shift_cells(augmentations: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for aug_type, severity_map in sorted(augmentations.items()):
        for severity, item in sorted(severity_map.items()):
            cells.append(
                {
                    "shift": aug_type,
                    "severity": severity,
                    "shifted_normal_mean": item.get("mean"),
                    "shifted_normal_median": item.get("median"),
                    "p25": item.get("p25"),
                    "p75": item.get("p75"),
                    "shifted_normal_fpr": item.get(
                        "shifted_normal_fpr",
                        item.get("fpr_over_clean_max"),
                    ),
                    "mean_score_shift": item.get("mean_score_shift"),
                    "median_score_shift": item.get("median_score_shift"),
                    "shifted_image_auroc": item.get(
                        "shifted_image_auroc",
                        item.get("image_auroc_vs_clean_anomaly"),
                    ),
                    "image_auroc_drop_from_clean": item.get(
                        "image_auroc_drop_from_clean"
                    ),
                }
            )
    return cells


def _aggregate_cell_metrics(
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    if not cells:
        return {}

    fpr_values = [c["shifted_normal_fpr"] for c in cells if c["shifted_normal_fpr"] is not None]
    auroc_drop_values = [
        c["image_auroc_drop_from_clean"]
        for c in cells
        if c["image_auroc_drop_from_clean"] is not None
    ]
    mean_shift_values = [c["mean_score_shift"] for c in cells if c["mean_score_shift"] is not None]
    worst_idx = max(
        range(len(cells)),
        key=lambda i: cells[i]["shifted_normal_fpr"] if cells[i]["shifted_normal_fpr"] is not None else -1.0,
    )
    worst = cells[worst_idx]

    severity_buckets: dict[str, dict[str, list[float]]] = {}
    for cell in cells:
        bucket = severity_buckets.setdefault(
            cell["severity"],
            {
                "shifted_normal_fpr": [],
                "image_auroc_drop_from_clean": [],
                "mean_score_shift": [],
            },
        )
        for key in bucket:
            value = cell[key]
            if value is not None:
                bucket[key].append(value)

    aggregate: dict[str, Any] = {
        "mean_shifted_normal_fpr": sum(fpr_values) / len(fpr_values) if fpr_values else None,
        "worst_shifted_normal_fpr": worst["shifted_normal_fpr"],
        "mean_fpr_over_clean_max": sum(fpr_values) / len(fpr_values) if fpr_values else None,
        "worst_fpr_over_clean_max": worst["shifted_normal_fpr"],
        "worst_shift_cell_by_fpr": f"{worst['shift']}/{worst['severity']}",
        "worst_cell": f"{worst['shift']}/{worst['severity']}",
        "mean_image_auroc_drop_from_clean": (
            sum(auroc_drop_values) / len(auroc_drop_values) if auroc_drop_values else None
        ),
        "worst_image_auroc_drop_from_clean": max(auroc_drop_values) if auroc_drop_values else None,
        "mean_score_shift": sum(mean_shift_values) / len(mean_shift_values) if mean_shift_values else None,
    }
    if auroc_drop_values:
        worst_auroc_drop = max(
            cells,
            key=lambda cell: cell["image_auroc_drop_from_clean"]
            if cell["image_auroc_drop_from_clean"] is not None
            else -1.0,
        )
        aggregate["worst_shift_cell_by_auroc_drop"] = (
            f"{worst_auroc_drop['shift']}/{worst_auroc_drop['severity']}"
        )
    for severity, bucket in severity_buckets.items():
        if bucket["shifted_normal_fpr"]:
            value = sum(bucket["shifted_normal_fpr"]) / len(bucket["shifted_normal_fpr"])
            aggregate[f"mean_shifted_normal_fpr_by_severity/{severity}"] = value
            aggregate[f"mean_fpr_by_severity/{severity}"] = value
        if bucket["image_auroc_drop_from_clean"]:
            aggregate[f"mean_image_auroc_drop_from_clean_by_severity/{severity}"] = (
                sum(bucket["image_auroc_drop_from_clean"])
                / len(bucket["image_auroc_drop_from_clean"])
            )
        if bucket["mean_score_shift"]:
            aggregate[f"mean_score_shift_by_severity/{severity}"] = (
                sum(bucket["mean_score_shift"]) / len(bucket["mean_score_shift"])
            )
    return aggregate


def _build_per_shift_metrics(cells: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for cell in cells:
        corruption = cell["shift"]
        severity = cell["severity"]
        metric_map = {
            f"shifted_image_auroc/{corruption}/{severity}": cell["shifted_image_auroc"],
            f"shifted_normal_fpr/{corruption}/{severity}": cell["shifted_normal_fpr"],
            f"image_auroc_drop_from_clean/{corruption}/{severity}": cell["image_auroc_drop_from_clean"],
            f"mean_score_shift/{corruption}/{severity}": cell["mean_score_shift"],
            f"median_score_shift/{corruption}/{severity}": cell["median_score_shift"],
        }
        for key, value in metric_map.items():
            if value is not None:
                metrics[key] = value
    return metrics


def log_summary_to_wandb(
    run, *, summary: dict[str, Any], summary_path: Path, log_path: Path
) -> None:
    if run is None:
        return

    metrics = summary.get("metrics", {})
    payload = summary.get("payload", {})
    augmentations = payload.get("augmentations", {})
    severity_spec_by_cell = payload.get("severity_spec_by_cell", {})

    top_line: dict[str, Any] = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and value is not None
    }

    cells = _build_shift_cells(augmentations)
    top_line.update(_aggregate_cell_metrics(cells))
    top_line.update(_build_per_shift_metrics(cells))
    run.log(top_line)

    import wandb  # noqa: WPS433

    if cells:
        shift_cells_table = wandb.Table(columns=list(SHIFT_CELL_COLUMNS))
        shift_summary_table = wandb.Table(columns=list(SHIFT_SUMMARY_TABLE_COLUMNS))
        for cell in cells:
            severity_param_raw = severity_spec_by_cell.get(
                f"{cell['shift']}/{cell['severity']}",
                payload.get("severity_spec", {}),
            )
            severity_param = json.dumps(severity_param_raw, ensure_ascii=True, sort_keys=True)
            shift_cells_table.add_data(
                cell["shift"],
                cell["severity"],
                cell["shifted_normal_mean"],
                cell["shifted_normal_median"],
                cell["p25"],
                cell["p75"],
                cell["shifted_normal_fpr"],
                cell["mean_score_shift"],
                cell["median_score_shift"],
                cell["shifted_image_auroc"],
                cell["image_auroc_drop_from_clean"],
                severity_param,
            )
            shift_summary_table.add_data(
                summary["baseline"],
                summary["class_name"],
                cell["shift"],
                cell["severity"],
                cell["shifted_normal_mean"],
                cell["shifted_normal_median"],
                cell["p25"],
                cell["p75"],
                cell["shifted_normal_fpr"],
                cell["shifted_image_auroc"],
                cell["image_auroc_drop_from_clean"],
                cell["mean_score_shift"],
                cell["median_score_shift"],
            )
        run.log(
            {
                "shift_cells": shift_cells_table,
                "shift_summary_table": shift_summary_table,
            }
        )

    artifact = wandb.Artifact(
        name=f"{summary['baseline'].lower()}-{summary['class_name']}-{summary['eval_type']}",
        type="condition-shift-summary",
    )
    if summary_path.exists():
        artifact.add_file(str(summary_path), name="summary.json")
    if log_path.exists():
        artifact.add_file(str(log_path), name="log.txt")
    run.log_artifact(artifact)


def log_preview_images_to_wandb(run, *, preview_images: dict[str, list[dict[str, Any]]]) -> None:
    if run is None or not preview_images:
        return

    import wandb  # noqa: WPS433

    logged_payload: dict[str, Any] = {}
    for key, items in preview_images.items():
        if not items:
            continue
        logged_payload[f"previews/{key}"] = [
            wandb.Image(item["image"], caption=item["caption"]) for item in items
        ]
    if logged_payload:
        run.log(logged_payload)


def finish_wandb_run(run) -> None:
    if run is None:
        return
    run.finish()
