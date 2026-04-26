"""Notebook-facing orchestration helpers for condition-shift runs."""

from __future__ import annotations

import importlib
import importlib.util
import site
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython.display import Markdown, display
except ModuleNotFoundError:  # pragma: no cover - used only in non-notebook local checks.
    class Markdown(str):
        pass

    def display(value: Any) -> None:
        print(value)


SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def now_string() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def display_title(title: str, body: str | None = None) -> None:
    text = f"## {title}"
    if body:
        text += f"\n\n{body}"
    display(Markdown(text))


def display_df(
    title: str,
    df: pd.DataFrame | None,
    *,
    empty_columns: list[str] | None = None,
    body: str | None = None,
) -> None:
    display_title(title, body)
    if df is None or df.empty:
        display(pd.DataFrame(columns=empty_columns or []))
    else:
        display(df)


def display_environment_summary(repo_root: Path, exp_root: Path, report_root: Path) -> None:
    display_title("Environment Summary")
    display(
        pd.DataFrame(
            [
                {"key": "REPO_ROOT", "value": str(repo_root)},
                {"key": "EXP_ROOT", "value": str(exp_root)},
                {"key": "REPORT_ROOT", "value": str(report_root)},
            ]
        )
    )


def ensure_importable_path(path: Path | str) -> None:
    path_str = str(Path(path).resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    importlib.invalidate_caches()
    for site_dir in site.getsitepackages():
        site.addsitedir(site_dir)
    user_site = site.getusersitepackages()
    if user_site:
        site.addsitedir(user_site)


def module_available(module_name: str, extra_paths: list[Path] | None = None) -> bool:
    for extra_path in extra_paths or []:
        ensure_importable_path(extra_path)
    return importlib.util.find_spec(module_name) is not None


def ordered_baselines(values: list[str] | pd.Series, preferred: list[str]) -> list[str]:
    seen = [value for value in dict.fromkeys(list(values)) if pd.notna(value)]
    return [baseline for baseline in preferred if baseline in seen] + [
        baseline for baseline in seen if baseline not in preferred
    ]


def draw_heatmap(
    ax: Any,
    pivot: pd.DataFrame,
    title: str,
    cmap: str = "Blues",
    center: float | None = None,
    value_fmt: str = "{:.2f}",
):
    if pivot is None or pivot.empty:
        ax.set_title(title)
        ax.axis("off")
        return None

    values = pivot.to_numpy(dtype=float)
    if center is None:
        im = ax.imshow(values, aspect="auto", cmap=cmap)
    else:
        vmax = np.nanmax(np.abs(values)) if np.size(values) else 1.0
        vmax = 1.0 if not np.isfinite(vmax) or vmax == 0 else vmax
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text = "-" if np.isnan(value) else value_fmt.format(value)
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8, color="black")
    return im


def display_run_plan(run_configs: list[dict[str, Any]]) -> None:
    rows = []
    for idx, config in enumerate(run_configs, start=1):
        rows.append(
            {
                "order": idx,
                "baseline": config["baseline"],
                "category": config["category"],
                "manifest_count": len(config["manifest_paths"]),
                "summary_name": config["summary_path"].name,
                "device": config["device"],
                "wandb": "on" if config.get("use_wandb") else "off",
            }
        )
    display(pd.DataFrame(rows))


def stream_subprocess(command: list[str], cwd: Path, show_output: bool = True) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def pump(pipe, prefix: str, sink: list[str]) -> None:
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                sink.append(line)
                if show_output:
                    print(f"{prefix} {line.rstrip()}", flush=True)
        finally:
            pipe.close()

    stdout_thread = threading.Thread(target=pump, args=(process.stdout, "[stdout]", stdout_lines))
    stderr_thread = threading.Thread(target=pump, args=(process.stderr, "[stderr]", stderr_lines))
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return {"returncode": returncode, "stdout": "".join(stdout_lines), "stderr": "".join(stderr_lines)}


def print_output_tail(name: str, text: str, tail_lines: int) -> None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        print(f"[{name}] <empty>")
        return
    tail = lines[-tail_lines:]
    print(f"[{name} tail] showing last {len(tail)} line(s)")
    for line in tail:
        print(line)


def build_baseline_specs(
    *,
    repo_root: Path,
    exp_root: Path,
    raw_loco_root: Path,
    univad_caption_data_root: Path,
    default_patchcore_device: str,
    univad_extra_args: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "PatchCore": {
            "runner_name": "PatchCore manifest shift evaluation",
            "runner_path": exp_root / "src" / "core" / "run_patchcore_manifest_shift.py",
            "runner_inputs": "category + manifest jsonl(s) + raw LOCO dataset root",
            "runner_outputs": "single summary json per category under reports/patchcore_manifest_shift",
            "report_subdir": "patchcore_manifest_shift",
            "external_dir": repo_root / "external" / "patchcore-inspection.clean",
            "external_repo_url": "https://github.com/amazon-science/patchcore-inspection.git",
            "data_root": raw_loco_root,
            "device": default_patchcore_device,
            "requires_cuda": False,
            "requires_checkpoints": False,
            "required_python_modules": [],
            "import_search_paths": [],
            "required_checkpoint_files": [],
            "checkpoint_download_urls": {},
            "required_data_paths": ["meta_not_required"],
            "required_mask_paths": [],
            "use_wandb": True,
            "wandb_group": "patchcore",
            "wandb_log_images": True,
            "wandb_max_images": 2,
            "extra_args": [],
            "notes": "PatchCore runner consumes raw LOCO directly and falls back to CPU automatically.",
        },
        "UniVAD": {
            "runner_name": "UniVAD manifest shift evaluation",
            "runner_path": exp_root / "src" / "univad" / "run_manifest_shift.py",
            "runner_inputs": "category + manifest jsonl(s) + MVTec-Caption style LOCO root",
            "runner_outputs": "single summary json per category under reports/univad_manifest_shift",
            "report_subdir": "univad_manifest_shift",
            "external_dir": repo_root / "external" / "UniVAD",
            "external_repo_url": "https://github.com/FantasticGNU/UniVAD.git",
            "data_root": univad_caption_data_root,
            "caption_prepare_script": exp_root / "src" / "univad" / "prepare_mvtec_loco.py",
            "groundingdino_dir": repo_root / "external" / "UniVAD" / "models" / "GroundingDINO",
            "checkpoint_root": repo_root / "external" / "UniVAD" / "pretrained_ckpts",
            "mask_root": repo_root / "external" / "UniVAD" / "masks" / "mvtec_loco_caption",
            "device": "cuda",
            "requires_cuda": True,
            "requires_checkpoints": True,
            "required_python_modules": ["groundingdino", "torchmetrics"],
            "import_search_paths": [repo_root / "external" / "UniVAD" / "models" / "GroundingDINO"],
            "required_local_paths": [
                repo_root / "external" / "UniVAD" / "models" / "dinov2" / "hubconf.py"
            ],
            "required_checkpoint_files": ["sam_hq_vit_h.pth", "groundingdino_swint_ogc.pth"],
            "checkpoint_download_urls": {
                "sam_hq_vit_h.pth": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
                "groundingdino_swint_ogc.pth": (
                    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
                    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
                ),
            },
            "required_data_paths": ["meta.json"],
            "required_mask_paths": [
                "train/good/000/grounding_mask.png",
                "test/good/000/grounding_mask.png",
                "test/logical_anomalies/000/grounding_mask.png",
                "test/structural_anomalies/000/grounding_mask.png",
            ],
            "use_wandb": True,
            "wandb_group": "univad",
            "wandb_log_images": True,
            "wandb_max_images": 2,
            "extra_args": univad_extra_args or ["--image-size", "224", "--k-shot", "1", "--round", "0"],
            "notes": (
                "UniVAD needs CUDA, recursive clone/submodules, MVTec-Caption style LOCO data, "
                "precomputed grounding masks, editable GroundingDINO, and local checkpoints."
            ),
        },
    }


def validate_baseline_names(names: list[str], *, specs: dict[str, Any], label: str) -> None:
    for baseline in names:
        if baseline not in specs:
            raise ValueError(f"Unknown {label} baseline={baseline}")


def resolve_requested_baselines(
    run_mode: str,
    run_baseline: str,
    sequential_baselines: list[str],
    *,
    specs: dict[str, Any],
) -> list[str]:
    if run_mode == "single":
        validate_baseline_names([run_baseline], specs=specs, label="single")
        return [run_baseline]
    if run_mode == "sequential":
        validate_baseline_names(sequential_baselines, specs=specs, label="sequential")
        return list(dict.fromkeys(sequential_baselines))
    raise ValueError(f"Unsupported RUN_MODE={run_mode}")


def discover_manifest_names(
    *,
    manifest_roots: list[Path],
    configured_names: list[str],
    auto_discover: bool,
    excluded_names: set[str],
) -> list[str]:
    if not auto_discover:
        return list(configured_names)
    discovered: list[str] = []
    for root in manifest_roots:
        if root.exists():
            discovered.extend(path.name for path in sorted(root.glob("query_*.jsonl")))
    return sorted({name for name in discovered if name not in excluded_names})


def resolve_manifest_paths(manifest_names: list[str], manifest_roots: list[Path]) -> list[Path]:
    manifest_paths: list[Path] = []
    for manifest_name in manifest_names:
        manifest_path = next(
            (root / manifest_name for root in manifest_roots if (root / manifest_name).exists()),
            None,
        )
        if manifest_path is None:
            searched = [str(root / manifest_name) for root in manifest_roots]
            raise FileNotFoundError(f"Manifest not found. searched={searched}")
        manifest_paths.append(manifest_path)
    return manifest_paths


def build_requested_run_configs(
    *,
    active_baselines: list[str],
    specs: dict[str, dict[str, Any]],
    categories: list[str],
    manifest_paths: list[Path],
    manifest_names: list[str],
    report_root: Path,
    wandb_project: str,
    wandb_mode: str,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for baseline in active_baselines:
        spec = specs[baseline]
        for category in categories:
            summary_path = report_root / spec["report_subdir"] / f"{category}_multi_all.json"
            log_path = report_root / spec["report_subdir"] / "logs" / f"{category}_multi_all.log.txt"
            runner_cmd = [
                sys.executable,
                str(spec["runner_path"]),
                "--category",
                category,
                "--manifest",
                *[str(path) for path in manifest_paths],
                "--data-root",
                str(spec["data_root"]),
                "--device",
                spec["device"],
                *spec["extra_args"],
            ]
            if spec["use_wandb"]:
                runner_cmd.extend(
                    [
                        "--use-wandb",
                        "--wandb-project",
                        wandb_project,
                        "--wandb-group",
                        spec["wandb_group"],
                        "--wandb-mode",
                        wandb_mode,
                    ]
                )
                if spec["wandb_log_images"]:
                    runner_cmd.extend(["--wandb-log-images", "--wandb-max-images", str(spec["wandb_max_images"])])
            configs.append(
                {
                    "baseline": baseline,
                    "category": category,
                    "device": spec["device"],
                    "data_root": str(spec["data_root"]),
                    "manifest_paths": list(manifest_paths),
                    "manifest_names": list(manifest_names),
                    "summary_path": summary_path,
                    "log_path": log_path,
                    "runner_cmd": runner_cmd,
                    "use_wandb": spec["use_wandb"],
                    "runner_name": spec["runner_name"],
                    "runner_path": spec["runner_path"],
                    "runner_inputs": spec["runner_inputs"],
                    "runner_outputs": spec["runner_outputs"],
                    "wandb_group": spec["wandb_group"],
                    "wandb_log_images": spec["wandb_log_images"],
                    "wandb_max_images": spec["wandb_max_images"],
                    "notes": spec["notes"],
                }
            )
    return configs


def display_experiment_preset(
    *,
    run_mode: str,
    requested_baselines: list[str],
    dashboard_baselines: list[str],
    categories: list[str],
    manifest_names: list[str],
    wandb_mode: str,
    stop_on_failure: bool,
    univad_flags: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    requested_run_configs: list[dict[str, Any]],
) -> None:
    display_title(
        "Experiment Preset",
        (
            f"RUN_MODE=`{run_mode}` | requested baselines=`{', '.join(requested_baselines)}` | "
            f"dashboard baselines=`{', '.join(dashboard_baselines)}`"
        ),
    )
    display(
        pd.DataFrame(
            [
                {
                    "run_mode": run_mode,
                    "requested_baselines": ", ".join(requested_baselines),
                    "dashboard_baselines": ", ".join(dashboard_baselines),
                    "category_count": len(categories),
                    "manifest_count": len(manifest_names),
                    "wandb_mode": wandb_mode,
                    "stop_on_failure": stop_on_failure,
                    **univad_flags,
                }
            ]
        )
    )
    display(
        pd.DataFrame(
            [
                {
                    "baseline": baseline,
                    "runner_name": specs[baseline]["runner_name"],
                    "data_root": str(specs[baseline]["data_root"]),
                    "device": specs[baseline]["device"],
                    "wandb_group": specs[baseline]["wandb_group"] if specs[baseline]["use_wandb"] else "off",
                    "wandb_mode": wandb_mode if specs[baseline]["use_wandb"] else "disabled",
                    "runner_outputs": specs[baseline]["runner_outputs"],
                    "notes": specs[baseline]["notes"],
                }
                for baseline in requested_baselines
            ]
        )
    )
    display(pd.DataFrame({"manifest_name": manifest_names}))
    display_run_plan(requested_run_configs)


def run_execution_queue(
    *,
    run_configs: list[dict[str, Any]],
    repo_root: Path,
    show_runner_output: bool,
    runner_output_tail_lines: int,
    stop_on_failure: bool,
) -> list[dict[str, Any]]:
    run_history: list[dict[str, Any]] = []
    total_runs = len(run_configs)
    if total_runs == 0:
        display_title("Execution Log", "No runnable jobs are queued. Check the readiness table above.")
        return run_history

    display_title("Execution Log", f"Started at `{now_string()}` with `{total_runs}` scheduled runs.")
    for idx, config in enumerate(run_configs, start=1):
        label = f"{config['baseline']} / {config['category']}"
        started_at = now_string()
        print("=" * 100)
        print(f"[{idx}/{total_runs}] START {label} @ {started_at}")
        print("summary_path =", config["summary_path"])
        print("log_path =", config["log_path"])
        print("command =", " ".join(config["runner_cmd"]))
        tic = time.perf_counter()
        result = stream_subprocess(config["runner_cmd"], cwd=repo_root, show_output=show_runner_output)
        elapsed_sec = round(time.perf_counter() - tic, 2)
        status = "success" if result["returncode"] == 0 else "failed"
        finished_at = now_string()
        summary_exists = Path(config["summary_path"]).exists()
        log_exists = Path(config["log_path"]).exists()
        run_history.append(
            {
                "order": idx,
                "baseline": config["baseline"],
                "category": config["category"],
                "status": status,
                "returncode": result["returncode"],
                "elapsed_sec": elapsed_sec,
                "started_at": started_at,
                "finished_at": finished_at,
                "summary_exists": summary_exists,
                "log_exists": log_exists,
                "summary_path": str(config["summary_path"]),
                "log_path": str(config["log_path"]),
            }
        )
        print(f"[{idx}/{total_runs}] END   {label} -> {status} ({elapsed_sec}s) @ {finished_at}")
        if result["returncode"] != 0:
            print("-" * 100)
            print(f"Failure recap for {label}")
            print_output_tail("stderr", result["stderr"], tail_lines=runner_output_tail_lines)
            if result["stdout"].strip():
                print_output_tail("stdout", result["stdout"], tail_lines=min(20, runner_output_tail_lines))
            if log_exists:
                print(f"runner log file = {config['log_path']}")
            if stop_on_failure:
                display(pd.DataFrame(run_history))
                raise RuntimeError(f"runner failed for {label}; inspect the streamed output above")

    print("=" * 100)
    display_title("Execution Summary", f"Finished at `{now_string()}`.")
    display(pd.DataFrame(run_history))
    failed_runs = [row for row in run_history if row["status"] == "failed"]
    if failed_runs and not stop_on_failure:
        display_title(
            "Execution Failures",
            f"`{len(failed_runs)}` run(s) failed. Successful summaries can still be compared below.",
        )
        display(pd.DataFrame(failed_runs))
    return run_history
