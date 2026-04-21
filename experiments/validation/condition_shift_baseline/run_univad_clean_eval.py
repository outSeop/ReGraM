from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from contracts import build_summary, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UniVAD clean evaluation and save a structured summary JSON."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="data/mvtec_loco_caption")
    parser.add_argument("--class-name", default="breakfast_box")
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/univad_clean_eval/breakfast_box.json",
    )
    parser.add_argument("--dataset", default="mvtec_loco")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_markdown_table(log_path: Path) -> list[dict[str, str]]:
    table_lines: list[str] = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line
        if "INFO:" in line:
            line = line.split("INFO:", 1)[1].strip()
        else:
            line = line.strip()
        if line.startswith("|"):
            table_lines.append(line)

    if len(table_lines) < 2:
        return []

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table_lines[1:]:
        stripped = line.strip("|")
        if not stripped:
            continue
        cells = [cell.strip() for cell in stripped.split("|")]
        if all(re.fullmatch(r"[-:]+", cell) for cell in cells):
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def to_percent_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def select_metric_rows(rows: list[dict[str, str]], class_name: str) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    object_key = class_name.replace("_", " ")
    class_row = next((row for row in rows if row.get("objects") == object_key), None)
    mean_row = next((row for row in rows if row.get("objects") == "mean"), None)
    return class_row, mean_row


def build_summary_payload(
    args: argparse.Namespace,
    repo_root: Path,
    output_path: Path,
    save_root: Path,
    log_path: Path,
    command: list[str],
) -> dict[str, object]:
    table_rows = parse_markdown_table(log_path)
    class_row, mean_row = select_metric_rows(table_rows, args.class_name)

    def metric_block(row: dict[str, str] | None) -> dict[str, object] | None:
        if row is None:
            return None
        return {
            "objects": row.get("objects"),
            "auroc_sp_percent": to_percent_float(row.get("auroc_sp")),
            "auroc_px_percent": to_percent_float(row.get("auroc_px")),
            "raw": row,
        }

    raw_payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "dataset": args.dataset,
        "class_name": args.class_name,
        "data_root": str((repo_root / args.data_root).resolve()),
        "device": args.device,
        "image_size": args.image_size,
        "k_shot": args.k_shot,
        "round": args.round,
        "save_root": str(save_root),
        "log_path": str(log_path),
        "command": command,
        "summary_path": str(output_path),
        "class_metrics": metric_block(class_row),
        "mean_metrics": metric_block(mean_row),
        "table_rows": table_rows,
    }
    return build_summary(
        baseline="UniVAD",
        dataset=args.dataset,
        class_name=args.class_name,
        eval_type="clean",
        device=args.device,
        output_path=output_path,
        log_path=log_path,
        config={
            "data_root": str((repo_root / args.data_root).resolve()),
            "image_size": args.image_size,
            "k_shot": args.k_shot,
            "round": args.round,
            "command": command,
        },
        metrics={
            "class_auroc_sp_percent": (raw_payload["class_metrics"] or {}).get("auroc_sp_percent"),
            "class_auroc_px_percent": (raw_payload["class_metrics"] or {}).get("auroc_px_percent"),
            "mean_auroc_sp_percent": (raw_payload["mean_metrics"] or {}).get("auroc_sp_percent"),
            "mean_auroc_px_percent": (raw_payload["mean_metrics"] or {}).get("auroc_px_percent"),
        },
        paths={
            "repo_root": repo_root,
            "data_root": (repo_root / args.data_root).resolve(),
            "save_root": save_root,
        },
        payload=raw_payload,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = (repo_root / args.output).resolve()
    save_root = output_path.parent
    univad_root = (repo_root / "external" / "UniVAD").resolve()
    data_root = (repo_root / args.data_root).resolve()

    os.environ.setdefault("TORCH_HOME", str(univad_root / "pretrained_ckpts" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    save_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(univad_root / "test_univad.py"),
        "--dataset",
        args.dataset,
        "--data_path",
        str(data_root),
        "--save_path",
        str(save_root),
        "--class_name",
        args.class_name,
        "--image_size",
        str(args.image_size),
        "--k_shot",
        str(args.k_shot),
        "--round",
        str(args.round),
        "--device",
        args.device,
    ]

    subprocess.run(command, cwd=univad_root, check=True)

    log_path = save_root / args.dataset / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"UniVAD log not found after clean eval: {log_path}")

    payload = build_summary_payload(
        args=args,
        repo_root=repo_root,
        output_path=output_path,
        save_root=save_root,
        log_path=log_path,
        command=command,
    )
    write_summary(payload, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
