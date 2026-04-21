"""Minimal wandb helpers for experiment tracking.

Role:
- optional tracking layer
- keep wandb integration out of the main runner logic
- no-op unless the runner explicitly enables wandb
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


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

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        tags=tags,
        config=config,
        mode=mode,
    )


def log_summary_to_wandb(run, *, summary: dict[str, Any], summary_path: Path, log_path: Path) -> None:
    if run is None:
        return

    metrics = summary.get("metrics", {})
    payload = summary.get("payload", {})
    augmentations = payload.get("augmentations", {})

    flat_metrics: dict[str, Any] = {
        "clean_image_auroc": metrics.get("clean_image_auroc"),
        "clean_good_mean": metrics.get("clean_good_mean"),
        "clean_good_fpr_over_clean_max": metrics.get("clean_good_fpr_over_clean_max"),
        "clean_anomaly_mean": metrics.get("clean_anomaly_mean"),
        "clean_anomaly_fpr_over_clean_max": metrics.get("clean_anomaly_fpr_over_clean_max"),
    }

    for aug_type, severity_map in augmentations.items():
        for severity, item in severity_map.items():
            prefix = f"{aug_type}/{severity}"
            flat_metrics[f"{prefix}/mean"] = item.get("mean")
            flat_metrics[f"{prefix}/fpr_over_clean_max"] = item.get("fpr_over_clean_max")
            flat_metrics[f"{prefix}/mean_score_shift"] = item.get("mean_score_shift")
            flat_metrics[f"{prefix}/image_auroc_vs_clean_anomaly"] = item.get("image_auroc_vs_clean_anomaly")

    run.log(flat_metrics)

    import wandb  # noqa: WPS433

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
