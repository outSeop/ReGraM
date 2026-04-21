from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "condition_shift_baseline.summary.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stringify_paths(payload: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        normalized[key] = str(value) if value is not None else ""
    return normalized


def build_summary(
    *,
    baseline: str,
    dataset: str,
    class_name: str,
    eval_type: str,
    device: str,
    output_path: Path,
    log_path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    paths: dict[str, Any],
    payload: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": utc_now_iso(),
        "baseline": baseline,
        "dataset": dataset,
        "class_name": class_name,
        "eval_type": eval_type,
        "device": device,
        "config": config,
        "metrics": metrics,
        "paths": stringify_paths(
            {
                **paths,
                "summary_path": output_path,
                "log_path": log_path,
            }
        ),
        "artifacts": stringify_paths(artifacts or {}),
        "payload": payload,
    }


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    ensure_parent(output_path)
    output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


def write_log(log_path: Path, lines: list[str]) -> None:
    ensure_parent(log_path)
    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
