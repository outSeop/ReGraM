"""Notebook dashboard loading and rendering helpers."""

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


import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from augmentation_runtime import apply_augmentation
from notebook_orchestration import Markdown, display, display_df, display_title, draw_heatmap, ordered_baselines
from repo_paths import REPO_ROOT


CLEAN_AUROC_METRICS = [
    ("clean_image_auroc", "overall"),
    ("clean_logical_image_auroc", "logical"),
    ("clean_structural_image_auroc", "structural"),
]
SHIFTED_AUROC_METRICS = [
    ("shifted_image_auroc", "overall"),
    ("shifted_image_auroc_vs_logical_anomaly", "logical"),
    ("shifted_image_auroc_vs_structural_anomaly", "structural"),
]
AUROC_DROP_METRICS = [
    ("image_auroc_drop_from_clean", "overall"),
    ("image_auroc_drop_from_clean_logical", "logical"),
    ("image_auroc_drop_from_clean_structural", "structural"),
]


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_dashboard_artifact_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(REPO_ROOT / path)
    return first_existing_path(candidates)


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def safe_max(values: list[float]) -> float:
    return float(max(values)) if values else np.nan


def metric_has_values(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def metric_split_long_df(
    df: pd.DataFrame,
    *,
    id_columns: list[str],
    metric_specs: list[tuple[str, str]],
    value_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=[*id_columns, "anomaly_scope", "metric_column", value_name])
    for metric_column, anomaly_scope in metric_specs:
        if metric_column not in df.columns:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    **{column: row.get(column) for column in id_columns},
                    "anomaly_scope": anomaly_scope,
                    "metric_column": metric_column,
                    value_name: row.get(metric_column),
                }
            )
    return pd.DataFrame(rows)


def normalize_patchcore_clean_eval(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    raw = payload.get("payload", payload)
    return [
        {
            "baseline": "PatchCore",
            "category": item.get("category"),
            "clean_metric_primary_name": "image_auroc",
            "clean_metric_primary": item.get("image_auroc"),
            "clean_metric_secondary_name": "full_pixel_auroc",
            "clean_metric_secondary": item.get("full_pixel_auroc"),
            "clean_metric_tertiary_name": "anomaly_pixel_auroc",
            "clean_metric_tertiary": item.get("anomaly_pixel_auroc"),
            "train_count": item.get("train_count"),
            "test_count": item.get("test_count"),
            "artifact_path": str(source_path),
        }
        for item in raw.get("categories", [])
    ]


def normalize_univad_clean_eval(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    metrics = payload.get("metrics", {})
    return [
        {
            "baseline": payload.get("baseline", "UniVAD"),
            "category": payload.get("class_name"),
            "clean_metric_primary_name": "class_auroc_sp_percent",
            "clean_metric_primary": metrics.get("class_auroc_sp_percent"),
            "clean_metric_secondary_name": "class_auroc_px_percent",
            "clean_metric_secondary": metrics.get("class_auroc_px_percent"),
            "clean_metric_tertiary_name": "mean_auroc_sp_percent",
            "clean_metric_tertiary": metrics.get("mean_auroc_sp_percent"),
            "train_count": None,
            "test_count": None,
            "artifact_path": str(source_path),
        }
    ]


def load_clean_eval_rows(baseline: str, category: str, report_root: Path) -> tuple[list[dict[str, Any]], Path | None]:
    if baseline == "PatchCore":
        candidates = [report_root / f"patchcore_clean_eval_{category}.json", report_root / "patchcore_clean_eval.json"]
    elif baseline == "UniVAD":
        candidates = [report_root / "univad_clean_eval" / f"{category}.json"]
    else:
        candidates = []
    source_path = first_existing_path(candidates)
    if source_path is None:
        return [], None
    payload = load_json_if_exists(source_path)
    if payload is None:
        return [], source_path
    if baseline == "PatchCore":
        return normalize_patchcore_clean_eval(payload, source_path), source_path
    if baseline == "UniVAD":
        return normalize_univad_clean_eval(payload, source_path), source_path
    return [], source_path


def load_identity_repro_rows(baseline: str, category: str, report_root: Path) -> tuple[list[dict[str, Any]], Path | None]:
    candidates = [report_root / "patchcore_identity_repro" / f"{category}.json"] if baseline == "PatchCore" else []
    source_path = first_existing_path(candidates)
    if source_path is None:
        return [], None
    payload = load_json_if_exists(source_path)
    if payload is None:
        return [], source_path
    score_diff = payload.get("score_diff", {})
    return [
        {
            "baseline": baseline,
            "category": payload.get("category", category),
            "identity_mean_abs_diff": score_diff.get("mean_abs_diff"),
            "identity_max_abs_diff": score_diff.get("max_abs_diff"),
            "identity_allclose_atol_1e6": score_diff.get("allclose_atol_1e-6"),
            "identity_allclose_atol_1e5": score_diff.get("allclose_atol_1e-5"),
            "direct_clean_mean": (payload.get("direct_clean") or {}).get("mean"),
            "manifest_identity_mean": (payload.get("manifest_identity") or {}).get("mean"),
            "artifact_path": str(source_path),
        }
    ], source_path


def load_manifest_identity_check_rows(report_root: Path) -> tuple[list[dict[str, Any]], Path | None]:
    source_path = first_existing_path([report_root / "identity_reproduction_check.json"])
    if source_path is None:
        return [], None
    payload = load_json_if_exists(source_path)
    if payload is None:
        return [], source_path
    return [
        {
            "manifest": payload.get("manifest", ""),
            "compared": payload.get("compared"),
            "mismatch_count": payload.get("mismatch_count"),
            "reproduced_exactly": payload.get("reproduced_exactly"),
            "artifact_path": str(source_path),
        }
    ], source_path


def load_dashboard_frames(
    *,
    summary_sources: list[dict[str, Any]],
    dashboard_baselines: list[str],
    categories: list[str],
    report_root: Path,
    severity_order: dict[str, int],
    top_k_worst_shifts: int = 10,
) -> dict[str, pd.DataFrame]:
    shift_rows: list[dict[str, Any]] = []
    clean_manifest_rows: list[dict[str, Any]] = []
    scoreboard_rows: list[dict[str, Any]] = []
    missing_summary_rows: list[dict[str, Any]] = []
    artifact_status_rows: list[dict[str, Any]] = []
    clean_eval_rows: list[dict[str, Any]] = []
    identity_repro_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    clean_reference_rows: list[dict[str, Any]] = []

    for config in summary_sources:
        summary_path = config["summary_path"]
        summary_exists = summary_path.exists()
        artifact_status_rows.append(
            {
                "baseline": config["baseline"],
                "category": config["category"],
                "artifact_type": "manifest_summary",
                "exists": summary_exists,
                "path": str(summary_path),
            }
        )
        if not summary_exists:
            missing_summary_rows.append(
                {
                    "baseline": config["baseline"],
                    "category": config["category"],
                    "summary_path": str(summary_path),
                    "log_path": str(config["log_path"]),
                }
            )
            continue

        summary = load_json_if_exists(summary_path)
        metrics = summary.get("metrics", {})
        payload = summary.get("payload", {})
        clean_good = payload.get("clean_good", {})
        clean_anomaly = payload.get("clean_anomaly", {})
        clean_logical_anomaly = payload.get("clean_logical_anomaly", {})
        clean_structural_anomaly = payload.get("clean_structural_anomaly", {})
        clean_scores = payload.get("clean_score_distributions", {})
        clean_normal_scores = [float(value) for value in clean_scores.get("clean_normal_scores", [])]
        clean_anomaly_scores = [float(value) for value in clean_scores.get("clean_anomaly_scores", [])]
        clean_logical_anomaly_scores = [
            float(value)
            for value in clean_scores.get("clean_logical_anomaly_scores", [])
        ]
        clean_structural_anomaly_scores = [
            float(value)
            for value in clean_scores.get("clean_structural_anomaly_scores", [])
        ]
        clean_image_auroc = metrics.get("clean_image_auroc")
        baseline = summary["baseline"]
        category = summary["class_name"]
        for sample_type, sample_rows in payload.get("clean_reference_maps", {}).items():
            for sample in sample_rows:
                clean_reference_rows.append(
                    {
                        "baseline": baseline,
                        "category": category,
                        "sample_type": sample.get("sample_type", sample_type),
                        "image_path": sample.get("image_path"),
                        "image_name": sample.get("image_name"),
                        "source_id": sample.get("source_id"),
                        "score": sample.get("score"),
                        "score_delta_from_clean_mean": sample.get("score_delta_from_clean_mean"),
                        "heatmap_path": sample.get("heatmap_path"),
                        "overlay_path": sample.get("overlay_path"),
                    }
                )

        clean_manifest_rows.append(
            {
                "baseline": baseline,
                "category": category,
                "clean_image_auroc": clean_image_auroc,
                "clean_good_mean": metrics.get("clean_good_mean", clean_good.get("mean")),
                "clean_good_median": metrics.get("clean_good_median", clean_good.get("median")),
                "clean_anomaly_mean": metrics.get("clean_anomaly_mean", clean_anomaly.get("mean")),
                "clean_anomaly_median": metrics.get("clean_anomaly_median", clean_anomaly.get("median")),
                "clean_logical_image_auroc": metrics.get(
                    "clean_logical_image_auroc",
                    payload.get("clean_logical_image_auroc"),
                ),
                "clean_logical_anomaly_mean": metrics.get(
                    "clean_logical_anomaly_mean",
                    clean_logical_anomaly.get("mean"),
                ),
                "clean_logical_anomaly_median": metrics.get(
                    "clean_logical_anomaly_median",
                    clean_logical_anomaly.get("median"),
                ),
                "clean_structural_image_auroc": metrics.get(
                    "clean_structural_image_auroc",
                    payload.get("clean_structural_image_auroc"),
                ),
                "clean_structural_anomaly_mean": metrics.get(
                    "clean_structural_anomaly_mean",
                    clean_structural_anomaly.get("mean"),
                ),
                "clean_structural_anomaly_median": metrics.get(
                    "clean_structural_anomaly_median",
                    clean_structural_anomaly.get("median"),
                ),
                "summary_path": str(summary_path),
            }
        )

        cell_rows: list[dict[str, Any]] = []
        severity_spec_by_cell = payload.get("severity_spec_by_cell", {})
        for aug_type, severity_map in payload.get("augmentations", {}).items():
            for severity, item in severity_map.items():
                cell_key = f"{aug_type}/{severity}"
                spec = severity_spec_by_cell.get(cell_key, payload.get("severity_spec", {}))
                shifted_image_auroc = item.get("shifted_image_auroc", item.get("image_auroc_vs_clean_anomaly"))
                image_auroc_drop_from_clean = item.get("image_auroc_drop_from_clean")
                if image_auroc_drop_from_clean is None and shifted_image_auroc is not None and clean_image_auroc is not None:
                    image_auroc_drop_from_clean = clean_image_auroc - shifted_image_auroc
                shifted_normal_fpr = item.get("shifted_normal_fpr", item.get("fpr_over_clean_max"))
                shifted_normal_scores = [float(value) for value in item.get("shifted_normal_scores", [])]
                shifted_image_auroc_vs_logical = item.get("shifted_image_auroc_vs_logical_anomaly")
                shifted_image_auroc_vs_structural = item.get("shifted_image_auroc_vs_structural_anomaly")
                row = {
                    "baseline": baseline,
                    "category": category,
                    "shift_family": aug_type,
                    "severity": severity,
                    "severity_rank": severity_order.get(severity, 999),
                    "cell_key": cell_key,
                    "shifted_normal_mean": item.get("shifted_normal_mean", item.get("mean")),
                    "shifted_normal_median": item.get("shifted_normal_median", item.get("median")),
                    "shifted_normal_p25": item.get("p25"),
                    "shifted_normal_p75": item.get("p75"),
                    "shifted_normal_fpr": shifted_normal_fpr,
                    "shifted_image_auroc": shifted_image_auroc,
                    "shifted_image_auroc_vs_logical_anomaly": shifted_image_auroc_vs_logical,
                    "shifted_image_auroc_vs_structural_anomaly": shifted_image_auroc_vs_structural,
                    "image_auroc_drop_from_clean": image_auroc_drop_from_clean,
                    "image_auroc_drop_from_clean_logical": item.get(
                        "image_auroc_drop_from_clean_logical"
                    ),
                    "image_auroc_drop_from_clean_structural": item.get(
                        "image_auroc_drop_from_clean_structural"
                    ),
                    "mean_score_shift": item.get("mean_score_shift"),
                    "median_score_shift": item.get("median_score_shift"),
                    "shifted_normal_scores": shifted_normal_scores,
                    **{f"severity_param_{key}": value for key, value in spec.items()},
                }
                shift_rows.append(row)
                cell_rows.append(row)
                distribution_rows.append(
                    {
                        "baseline": baseline,
                        "category": category,
                        "shift_family": aug_type,
                        "severity": severity,
                        "severity_rank": severity_order.get(severity, 999),
                        "clean_normal_scores": clean_normal_scores,
                        "clean_anomaly_scores": clean_anomaly_scores,
                        "clean_logical_anomaly_scores": clean_logical_anomaly_scores,
                        "clean_structural_anomaly_scores": clean_structural_anomaly_scores,
                        "shifted_normal_scores": shifted_normal_scores,
                    }
                )
                for sample in item.get("sample_rows", []):
                    gallery_rows.append(
                        {
                            "baseline": baseline,
                            "category": category,
                            "shift_family": aug_type,
                            "severity": severity,
                            "severity_rank": severity_order.get(severity, 999),
                            "image_path": sample.get("image_path"),
                            "image_name": sample.get("image_name"),
                            "source_id": sample.get("source_id"),
                            "score": sample.get("score"),
                            "score_delta_from_clean_mean": sample.get("score_delta_from_clean_mean"),
                            "augmentation_type": sample.get("augmentation_type", aug_type),
                            "seed": sample.get("seed"),
                            "params": json.dumps(sample.get("params", {}), ensure_ascii=True, sort_keys=True),
                            "params_dict": sample.get("params", {}),
                            "heatmap_path": sample.get("heatmap_path"),
                            "overlay_path": sample.get("overlay_path"),
                        }
                    )

        if cell_rows:
            fpr_values = [row["shifted_normal_fpr"] for row in cell_rows if pd.notna(row["shifted_normal_fpr"])]
            auroc_drop_values = [
                row["image_auroc_drop_from_clean"] for row in cell_rows if pd.notna(row["image_auroc_drop_from_clean"])
            ]
            mean_shift_values = [row["mean_score_shift"] for row in cell_rows if pd.notna(row["mean_score_shift"])]
            median_shift_values = [row["median_score_shift"] for row in cell_rows if pd.notna(row["median_score_shift"])]
            worst_fpr_cell = max(cell_rows, key=lambda row: row["shifted_normal_fpr"] if pd.notna(row["shifted_normal_fpr"]) else -1.0)
            worst_drop_cell = max(
                cell_rows,
                key=lambda row: row["image_auroc_drop_from_clean"] if pd.notna(row["image_auroc_drop_from_clean"]) else -1.0,
            )
            scoreboard_rows.append(
                {
                    "baseline": baseline,
                    "category": category,
                    "clean_image_auroc": clean_image_auroc,
                    "mean_shifted_normal_fpr": metrics.get("mean_shifted_normal_fpr", safe_mean(fpr_values)),
                    "worst_shifted_normal_fpr": metrics.get("worst_shifted_normal_fpr", safe_max(fpr_values)),
                    "worst_fpr_cell": metrics.get("worst_shift_cell_by_fpr", worst_fpr_cell.get("cell_key")),
                    "mean_image_auroc_drop_from_clean": metrics.get(
                        "mean_image_auroc_drop_from_clean", safe_mean(auroc_drop_values)
                    ),
                    "worst_image_auroc_drop_from_clean": metrics.get(
                        "worst_image_auroc_drop_from_clean", safe_max(auroc_drop_values)
                    ),
                    "worst_auroc_drop_cell": metrics.get("worst_shift_cell_by_auroc_drop", worst_drop_cell.get("cell_key")),
                    "mean_score_shift": metrics.get("mean_score_shift", safe_mean(mean_shift_values)),
                    "mean_median_score_shift": metrics.get("mean_median_score_shift", safe_mean(median_shift_values)),
                    "summary_path": str(summary_path),
                }
            )

    for baseline in dashboard_baselines:
        for category in categories:
            clean_rows, clean_source = load_clean_eval_rows(baseline, category, report_root)
            artifact_status_rows.append(
                {
                    "baseline": baseline,
                    "category": category,
                    "artifact_type": "clean_eval",
                    "exists": clean_source is not None,
                    "path": str(clean_source) if clean_source else "",
                }
            )
            clean_eval_rows.extend(clean_rows)

            identity_rows, identity_source = load_identity_repro_rows(baseline, category, report_root)
            artifact_status_rows.append(
                {
                    "baseline": baseline,
                    "category": category,
                    "artifact_type": "identity_repro",
                    "exists": identity_source is not None,
                    "path": str(identity_source) if identity_source else "",
                }
            )
            identity_repro_rows.extend(identity_rows)

    manifest_identity_check_rows, manifest_identity_source = load_manifest_identity_check_rows(report_root)
    artifact_status_rows.append(
        {
            "baseline": "shared",
            "category": "all",
            "artifact_type": "manifest_identity_check",
            "exists": manifest_identity_source is not None,
            "path": str(manifest_identity_source) if manifest_identity_source else "",
        }
    )

    shift_df = pd.DataFrame(shift_rows)
    if not shift_df.empty:
        shift_df = shift_df.sort_values(["baseline", "category", "shift_family", "severity_rank"]).reset_index(drop=True)
        corruption_summary_df = (
            shift_df.groupby(["baseline", "category", "shift_family"])
            .agg(
                mean_shifted_normal_fpr=("shifted_normal_fpr", "mean"),
                worst_shifted_normal_fpr=("shifted_normal_fpr", "max"),
                mean_image_auroc_drop_from_clean=("image_auroc_drop_from_clean", "mean"),
                worst_image_auroc_drop_from_clean=("image_auroc_drop_from_clean", "max"),
                mean_image_auroc_drop_from_clean_logical=("image_auroc_drop_from_clean_logical", "mean"),
                mean_image_auroc_drop_from_clean_structural=("image_auroc_drop_from_clean_structural", "mean"),
                mean_score_shift=("mean_score_shift", "mean"),
                mean_median_score_shift=("median_score_shift", "mean"),
            )
            .reset_index()
            .sort_values(["baseline", "category", "shift_family"])
            .reset_index(drop=True)
        )
        severity_table_df = shift_df[
            [
                "baseline",
                "category",
                "shift_family",
                "severity",
                "shifted_normal_mean",
                "shifted_normal_median",
                "shifted_normal_p25",
                "shifted_normal_p75",
                "shifted_normal_fpr",
                "shifted_image_auroc",
                "shifted_image_auroc_vs_logical_anomaly",
                "shifted_image_auroc_vs_structural_anomaly",
                "image_auroc_drop_from_clean",
                "image_auroc_drop_from_clean_logical",
                "image_auroc_drop_from_clean_structural",
                "mean_score_shift",
                "median_score_shift",
            ]
        ].copy()
        worst_fpr_ranking_df = shift_df.sort_values("shifted_normal_fpr", ascending=False).head(top_k_worst_shifts).reset_index(drop=True)
        worst_auroc_drop_ranking_df = (
            shift_df.sort_values("image_auroc_drop_from_clean", ascending=False).head(top_k_worst_shifts).reset_index(drop=True)
        )
        worst_mean_score_shift_ranking_df = (
            shift_df.sort_values("mean_score_shift", ascending=False).head(top_k_worst_shifts).reset_index(drop=True)
        )
    else:
        corruption_summary_df = pd.DataFrame()
        severity_table_df = pd.DataFrame()
        worst_fpr_ranking_df = pd.DataFrame()
        worst_auroc_drop_ranking_df = pd.DataFrame()
        worst_mean_score_shift_ranking_df = pd.DataFrame()

    clean_manifest_df = pd.DataFrame(clean_manifest_rows)
    if not clean_manifest_df.empty:
        clean_manifest_df = clean_manifest_df.sort_values(["baseline", "category"]).reset_index(drop=True)
    scoreboard_df = pd.DataFrame(scoreboard_rows)
    if not scoreboard_df.empty:
        scoreboard_df = scoreboard_df.sort_values(["baseline", "category"]).reset_index(drop=True)

    return {
        "shift_df": shift_df,
        "clean_manifest_df": clean_manifest_df,
        "scoreboard_df": scoreboard_df,
        "corruption_summary_df": corruption_summary_df,
        "severity_table_df": severity_table_df,
        "worst_fpr_ranking_df": worst_fpr_ranking_df,
        "worst_auroc_drop_ranking_df": worst_auroc_drop_ranking_df,
        "worst_mean_score_shift_ranking_df": worst_mean_score_shift_ranking_df,
        "missing_summary_df": pd.DataFrame(missing_summary_rows),
        "artifact_status_df": pd.DataFrame(artifact_status_rows),
        "clean_eval_df": pd.DataFrame(clean_eval_rows),
        "identity_repro_df": pd.DataFrame(identity_repro_rows),
        "manifest_identity_check_df": pd.DataFrame(manifest_identity_check_rows),
        "distribution_df": pd.DataFrame(distribution_rows),
        "gallery_df": pd.DataFrame(gallery_rows),
        "clean_reference_df": pd.DataFrame(clean_reference_rows),
    }


def display_dashboard_tables(frames: dict[str, pd.DataFrame], *, run_history: list[dict[str, Any]], dashboard_baselines: list[str], top_k: int) -> None:
    clean_manifest_df = frames["clean_manifest_df"]
    corruption_summary_df = frames["corruption_summary_df"]
    severity_table_df = frames["severity_table_df"]

    display_df("Artifact Coverage", frames["artifact_status_df"], empty_columns=["baseline", "category", "artifact_type", "exists", "path"])
    display_df(
        "Clean Summary Block",
        clean_manifest_df[
            [
                "baseline",
                "category",
                "clean_image_auroc",
                "clean_good_mean",
                "clean_good_median",
                "clean_anomaly_mean",
                "clean_anomaly_median",
                "clean_logical_image_auroc",
                "clean_logical_anomaly_mean",
                "clean_logical_anomaly_median",
                "clean_structural_image_auroc",
                "clean_structural_anomaly_mean",
                "clean_structural_anomaly_median",
            ]
        ]
        if not clean_manifest_df.empty
        else clean_manifest_df,
        empty_columns=["baseline", "category", "clean_image_auroc"],
    )
    if not clean_manifest_df.empty and clean_manifest_df["baseline"].nunique() > 1:
        for metric_column, anomaly_scope in CLEAN_AUROC_METRICS:
            display_df(
                f"Clean AUROC Comparison | {anomaly_scope}",
                clean_manifest_df.pivot(index="category", columns="baseline", values=metric_column).reset_index()
                if metric_column in clean_manifest_df.columns
                else pd.DataFrame(),
                empty_columns=["category"],
                body=(
                    f"Metric: `{metric_column}`."
                    if metric_has_values(clean_manifest_df, metric_column)
                    else f"Metric `{metric_column}` unavailable. Re-run 01 with updated runners for log/str split."
                ),
            )

    display_df(
        "Shift Summary Table",
        corruption_summary_df.rename(columns={"shift_family": "corruption_type"}),
        empty_columns=["baseline", "category", "corruption_type", "mean_shifted_normal_fpr"],
    )
    display_df(
        "Severity Table",
        severity_table_df.rename(columns={"shift_family": "corruption_type", "shifted_normal_p25": "p25", "shifted_normal_p75": "p75"}),
        empty_columns=["baseline", "category", "corruption_type", "severity", "shifted_normal_fpr"],
    )
    auroc_drop_split_df = metric_split_long_df(
        severity_table_df,
        id_columns=["baseline", "category", "shift_family", "severity"],
        metric_specs=AUROC_DROP_METRICS,
        value_name="auroc_drop_from_clean",
    )
    display_df(
        "Severity Table | AUROC Drop Split",
        auroc_drop_split_df.rename(columns={"shift_family": "corruption_type"}),
        empty_columns=["baseline", "category", "corruption_type", "severity", "anomaly_scope", "auroc_drop_from_clean"],
        body="Rows are separated into overall/logical/structural anomaly scopes.",
    )
    for title, key, metric_column in [
        ("Worst-Case Ranking | shifted_normal_fpr", "worst_fpr_ranking_df", "shifted_normal_fpr"),
        ("Worst-Case Ranking | mean_score_shift", "worst_mean_score_shift_ranking_df", "mean_score_shift"),
    ]:
        df = frames[key]
        display_df(
            title,
            df[["baseline", "category", "shift_family", "severity", metric_column, "shifted_image_auroc", "median_score_shift"]]
            if not df.empty
            else df,
            empty_columns=["baseline", "category", "shift_family", "severity", metric_column],
            body=f"Top {top_k} worst shift cells.",
        )
    for metric_column, anomaly_scope in AUROC_DROP_METRICS:
        ranking_df = (
            frames["shift_df"].sort_values(metric_column, ascending=False).head(top_k).reset_index(drop=True)
            if metric_has_values(frames["shift_df"], metric_column)
            else pd.DataFrame()
        )
        display_df(
            f"Worst-Case Ranking | AUROC drop | {anomaly_scope}",
            ranking_df[
                [
                    "baseline",
                    "category",
                    "shift_family",
                    "severity",
                    metric_column,
                    "shifted_image_auroc",
                    "shifted_image_auroc_vs_logical_anomaly",
                    "shifted_image_auroc_vs_structural_anomaly",
                ]
            ]
            if not ranking_df.empty
            else ranking_df,
            empty_columns=["baseline", "category", "shift_family", "severity", metric_column],
            body=(
                f"Top {top_k} worst shift cells for `{metric_column}`."
                if not ranking_df.empty
                else f"Metric `{metric_column}` unavailable. Re-run 01 with updated runners for log/str split."
            ),
        )

    display_df("Dedicated Clean Eval Results", frames["clean_eval_df"], empty_columns=["baseline", "category", "clean_metric_primary"])
    display_df("Identity Reproduction Check", frames["identity_repro_df"], empty_columns=["baseline", "category", "identity_max_abs_diff"])
    display_df(
        "Manifest Identity Validation",
        frames["manifest_identity_check_df"],
        empty_columns=["manifest", "compared", "mismatch_count", "reproduced_exactly"],
    )
    if not frames["missing_summary_df"].empty:
        display_df("Missing Summaries", frames["missing_summary_df"], empty_columns=["baseline", "category", "summary_path"])
    if run_history:
        display_df("Run History Snapshot", pd.DataFrame(run_history))

    display_df("Shift Robustness Headline", frames["scoreboard_df"], empty_columns=["baseline", "category", "mean_shifted_normal_fpr"])
    display_corruption_comparisons(corruption_summary_df, dashboard_baselines)
    display_summary_dashboard(corruption_summary_df)


def display_corruption_comparisons(corruption_summary_df: pd.DataFrame, dashboard_baselines: list[str]) -> None:
    if not corruption_summary_df.empty and corruption_summary_df["baseline"].nunique() > 1:
        reference_baseline = "PatchCore" if "PatchCore" in set(corruption_summary_df["baseline"]) else ordered_baselines(corruption_summary_df["baseline"], dashboard_baselines)[0]
        compare_frames = {
            "Corruption Comparison | Mean Shifted-Normal FPR": corruption_summary_df.pivot_table(
                index=["category", "shift_family"], columns="baseline", values="mean_shifted_normal_fpr", aggfunc="mean"
            ).reset_index(),
            "Corruption Comparison | Mean AUROC Drop": corruption_summary_df.pivot_table(
                index=["category", "shift_family"], columns="baseline", values="mean_image_auroc_drop_from_clean", aggfunc="mean"
            ).reset_index(),
            "Corruption Comparison | Mean Score Shift": corruption_summary_df.pivot_table(
                index=["category", "shift_family"], columns="baseline", values="mean_score_shift", aggfunc="mean"
            ).reset_index(),
        }
        for df in compare_frames.values():
            for baseline in [col for col in df.columns if col not in {"category", "shift_family"}]:
                if baseline != reference_baseline and reference_baseline in df.columns:
                    df[f"{baseline} - {reference_baseline}"] = df[baseline] - df[reference_baseline]
    else:
        compare_frames = {
            "Corruption Comparison | Mean Shifted-Normal FPR": pd.DataFrame(),
            "Corruption Comparison | Mean AUROC Drop": pd.DataFrame(),
            "Corruption Comparison | Mean Score Shift": pd.DataFrame(),
        }
    for title, df in compare_frames.items():
        display_df(title, df, empty_columns=["category", "shift_family"])


def display_summary_dashboard(corruption_summary_df: pd.DataFrame) -> None:
    if not corruption_summary_df.empty:
        summary_dashboard_df = corruption_summary_df.copy()
        summary_dashboard_df["risk_score"] = (
            summary_dashboard_df["mean_shifted_normal_fpr"].fillna(0.0)
            + summary_dashboard_df["mean_image_auroc_drop_from_clean"].fillna(0.0)
            + summary_dashboard_df["mean_score_shift"].fillna(0.0)
        )
        summary_dashboard_df = summary_dashboard_df.sort_values(
            ["risk_score", "worst_shifted_normal_fpr"], ascending=False
        ).reset_index(drop=True)
    else:
        summary_dashboard_df = pd.DataFrame()
    display_df(
        "Summary Dashboard",
        summary_dashboard_df,
        empty_columns=["baseline", "category", "shift_family", "mean_shifted_normal_fpr", "risk_score"],
    )


def display_wandb_panel_recommendations() -> None:
    display_title("Recommended W&B Panels", "Suggested W&B panels for the same notebook questions.")
    display(
        pd.DataFrame(
            [
                {
                    "panel_name": "Clean image AUROC across baselines",
                    "metric": "clean_image_auroc",
                    "group_by": "group (= baseline), config.class_name",
                },
                {
                    "panel_name": "Mean shifted-normal FPR",
                    "metric": "mean_shifted_normal_fpr",
                    "group_by": "group (= baseline), config.class_name",
                },
                {
                    "panel_name": "Worst shifted-normal FPR",
                    "metric": "worst_shifted_normal_fpr",
                    "group_by": "group (= baseline), config.class_name",
                },
                {
                    "panel_name": "Mean AUROC drop from clean",
                    "metric": "mean_image_auroc_drop_from_clean",
                    "group_by": "group (= baseline), config.class_name",
                },
                {
                    "panel_name": "Shift cells detail table",
                    "metric": "shift_cells (wandb.Table)",
                    "group_by": "native table: shift, severity",
                },
            ]
        )
    )


def _coerce_gallery_params(sample_row: dict[str, Any]) -> dict[str, Any]:
    params = sample_row.get("params_dict", {})
    if not isinstance(params, dict):
        raw_params = sample_row.get("params", "{}")
        try:
            params = json.loads(raw_params) if isinstance(raw_params, str) else dict(raw_params)
        except Exception:
            params = {}
    return json.loads(json.dumps(params, ensure_ascii=True))


def _apply_gallery_augmentation(original: Image.Image, sample_row: dict[str, Any]) -> Image.Image:
    params = _coerce_gallery_params(sample_row)
    augmentation_type = sample_row["augmentation_type"]
    seed = int(sample_row.get("seed", 0) or 0)
    return apply_augmentation(
        original.copy(),
        augmentation_type=augmentation_type,
        severity=sample_row["severity"],
        seed=seed,
        params=params,
    )


def build_gallery_panel(sample_row: dict[str, Any], panel_size: tuple[int, int] = (160, 160)) -> Image.Image:
    image_path = Path(sample_row["image_path"])
    with Image.open(image_path) as image_obj:
        original = image_obj.convert("RGB")
    augmented = _apply_gallery_augmentation(original, sample_row)
    original_thumb = original.resize(panel_size)
    augmented_thumb = augmented.resize(panel_size)
    canvas = Image.new("RGB", (panel_size[0] * 2, panel_size[1]), color=(255, 255, 255))
    canvas.paste(original_thumb, (0, 0))
    canvas.paste(augmented_thumb, (panel_size[0], 0))
    return canvas


def _load_resized_image(path_value: Any, panel_size: tuple[int, int]) -> Image.Image | None:
    path = resolve_dashboard_artifact_path(path_value)
    if path is None:
        return None
    with Image.open(path) as image_obj:
        return image_obj.convert("RGB").resize(panel_size)


def _set_image_axis(ax, image: Image.Image | None, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if image is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        return
    ax.imshow(image)


def render_clean_reference_gallery(clean_reference_df: pd.DataFrame, panel_size: tuple[int, int] = (160, 160)) -> None:
    if clean_reference_df.empty:
        return
    display_title(
        "Clean Reference Maps",
        "Top-scored clean normal, logical anomaly, and structural anomaly response maps.",
    )
    ordered_types = ["clean_normal", "logical_anomaly", "structural_anomaly"]
    for (baseline, category), group_df in clean_reference_df.groupby(["baseline", "category"]):
        display(Markdown(f"### {baseline} | {category} | clean/logical/structural reference"))
        sample_rows: list[dict[str, Any]] = []
        for sample_type in ordered_types:
            type_df = group_df[group_df["sample_type"] == sample_type]
            sample_rows.extend(type_df.sort_values("score", ascending=False).head(2).to_dict("records"))
        if not sample_rows:
            continue
        fig, axes = plt.subplots(
            len(sample_rows),
            3,
            figsize=(9, 2.8 * len(sample_rows)),
            constrained_layout=True,
        )
        if len(sample_rows) == 1:
            axes = np.asarray([axes])
        for row_idx, sample in enumerate(sample_rows):
            original = _load_resized_image(sample.get("image_path"), panel_size)
            heatmap = _load_resized_image(sample.get("heatmap_path"), panel_size)
            overlay = _load_resized_image(sample.get("overlay_path"), panel_size)
            title_prefix = f"{sample.get('sample_type')} | score={float(sample.get('score', 0.0)):.4f}"
            _set_image_axis(axes[row_idx, 0], original, f"{title_prefix}\nimage")
            _set_image_axis(axes[row_idx, 1], heatmap, "response map")
            _set_image_axis(axes[row_idx, 2], overlay, "overlay")
        plt.show()


def render_shifted_explainability_gallery(gallery_df: pd.DataFrame, panel_size: tuple[int, int] = (160, 160)) -> None:
    if gallery_df.empty:
        return
    display_title(
        "Shifted Normal Response Maps",
        "Top-scored shifted normal samples with clean source, shifted image, response map, and overlay.",
    )
    for (baseline, category, shift_family, severity), cell_df in gallery_df.groupby(
        ["baseline", "category", "shift_family", "severity"]
    ):
        display(Markdown(f"### {baseline} | {category} | {shift_family} | {severity}"))
        sample_rows = cell_df.sort_values("score", ascending=False).head(4).to_dict("records")
        fig, axes = plt.subplots(
            len(sample_rows),
            4,
            figsize=(12, 2.8 * max(len(sample_rows), 1)),
            constrained_layout=True,
        )
        if len(sample_rows) == 1:
            axes = np.asarray([axes])
        for row_idx, sample in enumerate(sample_rows):
            image_path = resolve_dashboard_artifact_path(sample["image_path"])
            if image_path is None:
                for col_idx, title in enumerate(["clean source", "shifted", "response map", "overlay"]):
                    _set_image_axis(axes[row_idx, col_idx], None, title)
                continue
            with Image.open(image_path) as image_obj:
                original = image_obj.convert("RGB")
            shifted = _apply_gallery_augmentation(original, sample)
            heatmap = _load_resized_image(sample.get("heatmap_path"), panel_size)
            overlay = _load_resized_image(sample.get("overlay_path"), panel_size)
            score = float(sample.get("score", 0.0))
            delta = float(sample.get("score_delta_from_clean_mean", 0.0))
            _set_image_axis(axes[row_idx, 0], original.resize(panel_size), f"clean source\n{sample.get('source_id')}")
            _set_image_axis(axes[row_idx, 1], shifted.resize(panel_size), f"shifted\nscore={score:.4f}")
            _set_image_axis(axes[row_idx, 2], heatmap, "response map")
            _set_image_axis(axes[row_idx, 3], overlay, f"overlay\ndelta={delta:.4f}")
        plt.show()


def render_explainability_gallery(gallery_df: pd.DataFrame, clean_reference_df: pd.DataFrame) -> None:
    render_clean_reference_gallery(clean_reference_df)
    render_shifted_explainability_gallery(gallery_df)


def render_visual_dashboard(frames: dict[str, pd.DataFrame], *, dashboard_baselines: list[str]) -> None:
    clean_manifest_df = frames["clean_manifest_df"]
    shift_df = frames["shift_df"]
    distribution_df = frames["distribution_df"]
    gallery_df = frames["gallery_df"]
    clean_reference_df = frames.get("clean_reference_df", pd.DataFrame())

    if clean_manifest_df.empty and shift_df.empty:
        display_title("Visual Dashboard", "No result rows are available yet.")
        return
    display_title(
        "Visual Dashboard",
        "Clean baseline, log/str anomaly split, corruption trend, score distributions, and response-map gallery.",
    )

    if not clean_manifest_df.empty:
        render_clean_auroc_split_plots(clean_manifest_df, dashboard_baselines)

    if not shift_df.empty:
        render_shift_plots(shift_df, dashboard_baselines)
    if not distribution_df.empty:
        render_distribution_plots(distribution_df)
    if not gallery_df.empty or not clean_reference_df.empty:
        render_explainability_gallery(gallery_df, clean_reference_df)


def render_clean_auroc_split_plots(clean_manifest_df: pd.DataFrame, dashboard_baselines: list[str]) -> None:
    clean_plot_df = metric_split_long_df(
        clean_manifest_df,
        id_columns=["baseline", "category"],
        metric_specs=CLEAN_AUROC_METRICS,
        value_name="clean_image_auroc",
    )
    baselines_in_clean = ordered_baselines(clean_manifest_df["baseline"], dashboard_baselines)
    categories_in_clean = list(dict.fromkeys(clean_manifest_df["category"]))
    x = np.arange(len(categories_in_clean))
    width = 0.8 / max(len(baselines_in_clean), 1)
    fig, axes = plt.subplots(1, len(CLEAN_AUROC_METRICS), figsize=(6 * len(CLEAN_AUROC_METRICS), 4.5), constrained_layout=True)
    if len(CLEAN_AUROC_METRICS) == 1:
        axes = [axes]
    for ax, (_, anomaly_scope) in zip(axes, CLEAN_AUROC_METRICS):
        scope_df = clean_plot_df[clean_plot_df["anomaly_scope"] == anomaly_scope]
        if scope_df["clean_image_auroc"].notna().any():
            for idx, baseline in enumerate(baselines_in_clean):
                baseline_df = scope_df[scope_df["baseline"] == baseline].set_index("category")
                values = [
                    baseline_df.loc[category, "clean_image_auroc"] if category in baseline_df.index else np.nan
                    for category in categories_in_clean
                ]
                ax.bar(x + idx * width - ((len(baselines_in_clean) - 1) * width / 2), values, width=width, label=baseline)
            ax.set_xticks(x)
            ax.set_xticklabels(categories_in_clean, rotation=0)
            ax.set_ylabel("AUROC (%)")
            ax.legend(title="baseline")
        else:
            ax.text(0.5, 0.5, "split metric unavailable\nre-run 01", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"Clean Image AUROC | {anomaly_scope}")
    plt.show()


def render_shift_plots(shift_df: pd.DataFrame, dashboard_baselines: list[str]) -> None:
    shifted_auroc_long_df = metric_split_long_df(
        shift_df,
        id_columns=["baseline", "category", "shift_family", "severity", "severity_rank"],
        metric_specs=SHIFTED_AUROC_METRICS,
        value_name="shifted_image_auroc",
    )
    shifted_auroc_bar_df = (
        shifted_auroc_long_df.groupby(["baseline", "category", "anomaly_scope", "shift_family", "severity", "severity_rank"])
        .agg(shifted_image_auroc=("shifted_image_auroc", "mean"))
        .reset_index()
        .sort_values(["baseline", "category", "anomaly_scope", "shift_family", "severity_rank"])
    )
    for (baseline, category), plot_df in shifted_auroc_bar_df.groupby(["baseline", "category"]):
        fig, axes = plt.subplots(1, len(SHIFTED_AUROC_METRICS), figsize=(6 * len(SHIFTED_AUROC_METRICS), 4.8), constrained_layout=True)
        if len(SHIFTED_AUROC_METRICS) == 1:
            axes = [axes]
        for ax, (_, anomaly_scope) in zip(axes, SHIFTED_AUROC_METRICS):
            scope_df = plot_df[plot_df["anomaly_scope"] == anomaly_scope]
            families = list(dict.fromkeys(scope_df["shift_family"]))
            severities = [severity for severity in ["low", "medium", "high"] if severity in set(scope_df["severity"])]
            x = np.arange(len(families))
            width = 0.8 / max(len(severities), 1)
            if scope_df["shifted_image_auroc"].notna().any():
                for idx, severity in enumerate(severities):
                    severity_df = scope_df[scope_df["severity"] == severity].set_index("shift_family")
                    values = [
                        severity_df.loc[family, "shifted_image_auroc"] if family in severity_df.index else np.nan
                        for family in families
                    ]
                    ax.bar(x + idx * width - ((len(severities) - 1) * width / 2), values, width=width, label=severity)
                ax.set_xticks(x)
                ax.set_xticklabels(families, rotation=25, ha="right")
                ax.set_ylabel("shifted image AUROC (%)")
                ax.legend(title="severity")
            else:
                ax.text(0.5, 0.5, "split metric unavailable\nre-run 01", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(f"{baseline} | {category} | shifted AUROC | {anomaly_scope}")
        plt.show()

    severity_line_df = (
        shift_df.groupby(["baseline", "category", "shift_family", "severity", "severity_rank"])
        .agg(
            shifted_normal_fpr=("shifted_normal_fpr", "mean"),
            image_auroc_drop_from_clean=("image_auroc_drop_from_clean", "mean"),
            image_auroc_drop_from_clean_logical=("image_auroc_drop_from_clean_logical", "mean"),
            image_auroc_drop_from_clean_structural=("image_auroc_drop_from_clean_structural", "mean"),
            mean_score_shift=("mean_score_shift", "mean"),
            median_score_shift=("median_score_shift", "mean"),
        )
        .reset_index()
        .sort_values(["baseline", "category", "shift_family", "severity_rank"])
    )
    for (baseline, category, shift_family), family_df in severity_line_df.groupby(["baseline", "category", "shift_family"]):
        fig, axes = plt.subplots(2, 3, figsize=(18, 7.5), constrained_layout=True)
        axes_flat = axes.ravel()
        for ax, (column, title, ylabel) in zip(
            axes_flat,
            [
                ("shifted_normal_fpr", "shifted_normal_fpr", "FPR"),
                ("image_auroc_drop_from_clean", "AUROC drop | overall", "AUROC drop"),
                ("image_auroc_drop_from_clean_logical", "AUROC drop | logical", "AUROC drop"),
                ("image_auroc_drop_from_clean_structural", "AUROC drop | structural", "AUROC drop"),
                ("mean_score_shift", "mean_score_shift", "mean score shift"),
                ("median_score_shift", "median_score_shift", "median score shift"),
            ],
        ):
            if family_df[column].notna().any():
                ax.plot(family_df["severity"], family_df[column], marker="o", linewidth=2)
            else:
                ax.text(0.5, 0.5, "metric unavailable\nre-run 01", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(f"{shift_family} | {title}")
            ax.set_xlabel("severity")
            ax.set_ylabel(ylabel)
        plt.show()

    baselines_in_shift = ordered_baselines(shift_df["baseline"], dashboard_baselines)
    for metric_name, title, cmap in [
        ("shifted_normal_fpr", "Shifted-Normal FPR heatmap", "YlOrRd"),
        ("image_auroc_drop_from_clean", "AUROC-drop heatmap", "YlOrBr"),
        ("image_auroc_drop_from_clean_logical", "Logical AUROC-drop heatmap", "YlOrBr"),
        ("image_auroc_drop_from_clean_structural", "Structural AUROC-drop heatmap", "YlOrBr"),
        ("median_score_shift", "Median score-shift heatmap", "PuBuGn"),
    ]:
        fig, axes = plt.subplots(1, len(baselines_in_shift), figsize=(6 * len(baselines_in_shift), 5), constrained_layout=True)
        if len(baselines_in_shift) == 1:
            axes = [axes]
        for ax, baseline in zip(axes, baselines_in_shift):
            baseline_df = shift_df[shift_df["baseline"] == baseline]
            pivot = baseline_df.pivot_table(index="shift_family", columns="severity", values=metric_name, aggfunc="mean")
            ordered_cols = [column for column in ["low", "medium", "high"] if column in pivot.columns]
            pivot = pivot[ordered_cols]
            im = draw_heatmap(ax, pivot, f"{baseline} | {title}", cmap=cmap, value_fmt="{:.2f}")
            if im is not None:
                plt.colorbar(im, ax=ax, shrink=0.8)
        plt.show()


def render_distribution_plots(distribution_df: pd.DataFrame) -> None:
    display_title(
        "Distribution Visualization",
        "Compare clean normal, shifted normal, logical anomaly, and structural anomaly score distributions.",
    )
    for (baseline, category, shift_family), family_df in distribution_df.groupby(["baseline", "category", "shift_family"]):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
        boxplot_data = []
        boxplot_labels = []
        clean_scores = family_df.iloc[0]["clean_normal_scores"]
        if clean_scores:
            boxplot_data.append(clean_scores)
            boxplot_labels.append("clean")
        for _, row in family_df.sort_values("severity_rank").iterrows():
            shifted_scores = row["shifted_normal_scores"]
            if shifted_scores:
                boxplot_data.append(shifted_scores)
                boxplot_labels.append(row["severity"])
        if boxplot_data:
            axes[0].boxplot(boxplot_data, labels=boxplot_labels, showfliers=False)
            axes[0].set_title(f"{baseline} | {shift_family} | boxplot")
            axes[0].set_ylabel("anomaly score")
        else:
            axes[0].set_title(f"{baseline} | {shift_family} | boxplot")
            axes[0].text(0.5, 0.5, "No score lists in summary. Re-run with updated runner.", ha="center", va="center")
            axes[0].axis("off")

        if clean_scores:
            axes[1].hist(clean_scores, bins=20, alpha=0.35, label="clean normal", density=True)
        clean_logical_anomaly_scores = family_df.iloc[0].get("clean_logical_anomaly_scores", [])
        clean_structural_anomaly_scores = family_df.iloc[0].get("clean_structural_anomaly_scores", [])
        clean_anomaly_scores = family_df.iloc[0]["clean_anomaly_scores"]
        if clean_logical_anomaly_scores:
            axes[1].hist(
                clean_logical_anomaly_scores,
                bins=20,
                histtype="step",
                linewidth=2,
                label="clean logical anomaly",
                density=True,
            )
        if clean_structural_anomaly_scores:
            axes[1].hist(
                clean_structural_anomaly_scores,
                bins=20,
                histtype="step",
                linewidth=2,
                label="clean structural anomaly",
                density=True,
            )
        if clean_anomaly_scores and not (
            clean_logical_anomaly_scores or clean_structural_anomaly_scores
        ):
            axes[1].hist(
                clean_anomaly_scores,
                bins=20,
                histtype="step",
                linewidth=2,
                label="clean anomaly",
                density=True,
            )
        for _, row in family_df.sort_values("severity_rank").iterrows():
            shifted_scores = row["shifted_normal_scores"]
            if shifted_scores:
                axes[1].hist(shifted_scores, bins=20, histtype="step", linewidth=2, label=f"shifted normal | {row['severity']}", density=True)
        axes[1].set_title(f"{baseline} | {shift_family} | histogram overlay")
        axes[1].set_xlabel("anomaly score")
        axes[1].set_ylabel("density")
        axes[1].legend(fontsize=8)
        plt.show()


def render_gallery(gallery_df: pd.DataFrame) -> None:
    render_shifted_explainability_gallery(gallery_df)
