"""Shared path and phase-logging utilities for condition_shift_baseline runners."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "index.md").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from: {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]


def now_kst_string() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def log_phase(log_lines: list[str], phase: str, message: str) -> None:
    line = f"[{now_kst_string()}] {phase}: {message}"
    print(line, flush=True)
    log_lines.append(line)


def finish_phase(
    log_lines: list[str], phase: str, started_at: float, extra: str = ""
) -> None:
    elapsed = time.perf_counter() - started_at
    suffix = f" ({elapsed:.2f}s)"
    if extra:
        suffix += f" | {extra}"
    log_phase(log_lines, phase, f"done{suffix}")
