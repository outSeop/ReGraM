"""Notebook-facing orchestration helpers for condition-shift runs."""

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


import importlib
import importlib.util
import site
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
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
SEVERITY_ALIASES = {"middle": "medium", "mid": "medium"}
MODEL_CONFIG_KEY_ALIASES = {"patchcore": "PatchCore", "univad": "UniVAD"}
MODEL_CONFIG_EXTRA_KEYS = {"extra_args"}
DEFAULT_MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "PatchCore": {
        "resize": 256,
        "imagesize": 224,
        "batch_size": 1,
        "num_workers": 0,
        "sampler_percentage": 0.001,
        "export_heatmaps": True,
        "heatmap_max_images": 2,
        "extra_args": [],
    },
    "UniVAD": {
        "image_size": 224,
        "k_shot": 1,
        "round": 0,
        "export_heatmaps": True,
        "heatmap_max_images": 2,
        "extra_args": [],
    },
}


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


def load_experiment_config(config_path: Path | str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load experiment YAML configs. "
            "Install it in the runtime with `pip install pyyaml`, or run in Colab where it is usually available."
        ) from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Experiment config must be a YAML mapping: {path}")
    return loaded


def default_model_config() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULT_MODEL_CONFIG)


def canonical_model_name(name: str) -> str:
    return MODEL_CONFIG_KEY_ALIASES.get(str(name).lower(), str(name))


def resolve_model_config(model_config: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    resolved = default_model_config()
    if model_config is None:
        return resolved
    for baseline, config in model_config.items():
        canonical_baseline = canonical_model_name(str(baseline))
        if canonical_baseline not in resolved:
            raise ValueError(f"Unknown model_config baseline={baseline}")
        if config is None:
            continue
        if not isinstance(config, dict):
            raise TypeError(f"model_config[{baseline!r}] must be a dict")
        resolved[canonical_baseline].update(config)
        validate_single_model_config(resolved[canonical_baseline], baseline=canonical_baseline)
    return resolved


def model_config_from_experiment_config(experiment_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = experiment_config.get("models", {})
    if not isinstance(models, dict):
        raise TypeError("experiment config `models` must be a mapping")
    return resolve_model_config(models)


def model_sweep_from_experiment_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    sweep = experiment_config.get("sweep", {})
    if sweep is None:
        return {}
    if not isinstance(sweep, dict):
        raise TypeError("experiment config `sweep` must be a mapping")
    return sweep


def validate_single_model_config(config: dict[str, Any], *, baseline: str) -> None:
    for key, value in config.items():
        if key in MODEL_CONFIG_EXTRA_KEYS:
            continue
        if isinstance(value, (list, tuple)):
            raise ValueError(
                f"List-valued model config is not allowed at models.{baseline}.{key}. "
                "Put multiple values under `sweep.models.<baseline>` instead."
            )
        if isinstance(value, dict):
            raise ValueError(
                f"Nested model config is not allowed at models.{baseline}.{key}. "
                "Use scalar values, or put grid values under `sweep.models.<baseline>`."
            )


def format_model_config(config: dict[str, Any]) -> str:
    parts = []
    for key, value in config.items():
        if key == "extra_args" and not value:
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def model_variant_display_name(baseline: str, variant_id: str) -> str:
    return baseline if variant_id == "default" else f"{baseline}:{variant_id}"


def build_model_variant_id(values: dict[str, Any]) -> str:
    if not values:
        return "default"
    parts = []
    for key, value in values.items():
        value_text = str(value).replace(".", "p").replace("-", "m")
        safe_value = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value_text)
        parts.append(f"{key}_{safe_value}")
    return "__".join(parts)


def expand_model_variants(
    resolved_model_config: dict[str, dict[str, Any]],
    model_sweep: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    sweep = model_sweep or {}
    enabled = bool(sweep.get("enabled", False))
    sweep_models = sweep.get("models", {}) if enabled else {}
    if sweep_models is None:
        sweep_models = {}
    if not isinstance(sweep_models, dict):
        raise TypeError("sweep.models must be a mapping")

    canonical_sweep_models = {
        canonical_model_name(str(baseline)): config
        for baseline, config in sweep_models.items()
    }
    for baseline in canonical_sweep_models:
        if baseline not in resolved_model_config:
            raise ValueError(f"Unknown sweep model baseline={baseline}")
    variants_by_baseline: dict[str, list[dict[str, Any]]] = {}
    for baseline, base_config in resolved_model_config.items():
        sweep_config = canonical_sweep_models.get(baseline)
        if not sweep_config:
            variants_by_baseline[baseline] = [
                {
                    "id": "default",
                    "label": "default",
                    "output_subdir": None,
                    "sweep_values": {},
                    "config": deepcopy(base_config),
                }
            ]
            continue
        if not isinstance(sweep_config, dict):
            raise TypeError(f"sweep.models.{baseline} must be a mapping")

        keys = list(sweep_config.keys())
        value_grid: list[list[Any]] = []
        for key in keys:
            values = sweep_config[key]
            if isinstance(values, (list, tuple)):
                value_list = list(values)
            else:
                value_list = [values]
            if not value_list:
                raise ValueError(f"sweep.models.{baseline}.{key} cannot be empty")
            value_grid.append(value_list)

        variants: list[dict[str, Any]] = []
        for values in product(*value_grid):
            sweep_values = dict(zip(keys, values))
            variant_config = deepcopy(base_config)
            variant_config.update(sweep_values)
            validate_single_model_config(variant_config, baseline=baseline)
            variant_id = build_model_variant_id(sweep_values)
            variants.append(
                {
                    "id": variant_id,
                    "label": ", ".join(f"{key}={value}" for key, value in sweep_values.items()),
                    "output_subdir": variant_id,
                    "sweep_values": sweep_values,
                    "config": variant_config,
                }
            )
        variants_by_baseline[baseline] = variants
    return variants_by_baseline


def build_patchcore_extra_args(config: dict[str, Any]) -> list[str]:
    args = [
        "--resize",
        str(config["resize"]),
        "--imagesize",
        str(config["imagesize"]),
        "--batch-size",
        str(config["batch_size"]),
        "--num-workers",
        str(config["num_workers"]),
        "--sampler-percentage",
        str(config["sampler_percentage"]),
    ]
    if config.get("export_heatmaps", False):
        args.extend(["--export-heatmaps", "--heatmap-max-images", str(config["heatmap_max_images"])])
    args.extend(str(value) for value in config.get("extra_args", []))
    return args


def build_univad_extra_args(config: dict[str, Any]) -> list[str]:
    args = [
        "--image-size",
        str(config["image_size"]),
        "--k-shot",
        str(config["k_shot"]),
        "--round",
        str(config["round"]),
    ]
    if config.get("export_heatmaps", False):
        args.extend(["--export-heatmaps", "--heatmap-max-images", str(config["heatmap_max_images"])])
    args.extend(str(value) for value in config.get("extra_args", []))
    return args


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
                "display_baseline": config.get("display_baseline", config["baseline"]),
                "model_variant": config.get("model_variant_id", "default"),
                "category": config["category"],
                "manifest_count": len(config["manifest_paths"]),
                "severities": ", ".join(config.get("selected_severities", [])) or "all",
                "summary_name": config["summary_path"].name,
                "device": config["device"],
                "model_config": format_model_config(config.get("model_config", {})),
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
    model_config: dict[str, dict[str, Any]] | None = None,
    model_sweep: dict[str, Any] | None = None,
    patchcore_extra_args: list[str] | None = None,
    univad_extra_args: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    resolved_model_config = resolve_model_config(model_config)
    patchcore_config = resolved_model_config["PatchCore"]
    univad_config = resolved_model_config["UniVAD"]
    model_variants = expand_model_variants(resolved_model_config, model_sweep)
    patchcore_variants = [
        {
            **variant,
            "extra_args": patchcore_extra_args
            if patchcore_extra_args is not None
            else build_patchcore_extra_args(variant["config"]),
        }
        for variant in model_variants["PatchCore"]
    ]
    univad_variants = [
        {
            **variant,
            "extra_args": univad_extra_args
            if univad_extra_args is not None
            else build_univad_extra_args(variant["config"]),
        }
        for variant in model_variants["UniVAD"]
    ]
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
            "model_config": patchcore_config,
            "model_variants": patchcore_variants,
            "extra_args": patchcore_extra_args
            if patchcore_extra_args is not None
            else build_patchcore_extra_args(patchcore_config),
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
                "groundingdino_swint_ogc.pth": [
                    (
                        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
                        "v0.1.0-alpha/groundingdino_swint_ogc.pth"
                    ),
                    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
                ],
            },
            "checkpoint_expected_bytes": {
                "sam_hq_vit_h.pth": 2_570_940_653,
                "groundingdino_swint_ogc.pth": 693_997_677,
            },
            "checkpoint_sha256": {
                "sam_hq_vit_h.pth": "a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35",
                "groundingdino_swint_ogc.pth": "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
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
            "model_config": univad_config,
            "model_variants": univad_variants,
            "extra_args": univad_extra_args
            if univad_extra_args is not None
            else build_univad_extra_args(univad_config),
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
    selected_names = [name for name in configured_names if name not in excluded_names]
    if selected_names:
        return selected_names
    if not auto_discover:
        return []
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


def normalize_selected_severities(severities: list[str] | tuple[str, ...] | None) -> list[str]:
    if not severities:
        return []
    normalized: list[str] = []
    for raw_value in severities:
        value = str(raw_value).strip().lower()
        if not value or value == "all":
            return []
        value = SEVERITY_ALIASES.get(value, value)
        if value not in SEVERITY_ORDER:
            raise ValueError(
                f"Unsupported severity: {raw_value}. "
                "Use one or more of: low, medium, high. Alias: middle -> medium."
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


def derive_run_output_suffix(manifest_names: list[str], selected_severities: list[str]) -> str:
    if len(manifest_names) == 1:
        manifest_name = manifest_names[0]
        if manifest_name.startswith("query_") and manifest_name.endswith(".jsonl"):
            shift_family = manifest_name[len("query_") : -len(".jsonl")]
        else:
            shift_family = Path(manifest_name).stem
    else:
        shift_family = "multi"
    severity_label = selected_severities[0] if len(selected_severities) == 1 else "multi" if selected_severities else "all"
    return f"{shift_family}_{severity_label}"


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
    selected_severities: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    normalized_severities = normalize_selected_severities(selected_severities)
    output_suffix = derive_run_output_suffix(manifest_names, normalized_severities)
    for baseline in active_baselines:
        spec = specs[baseline]
        for variant in spec.get(
            "model_variants",
            [
                {
                    "id": "default",
                    "label": "default",
                    "output_subdir": None,
                    "sweep_values": {},
                    "config": dict(spec.get("model_config", {})),
                    "extra_args": list(spec.get("extra_args", [])),
                }
            ],
        ):
            variant_id = str(variant.get("id", "default"))
            variant_report_subdir = Path(spec["report_subdir"])
            if variant.get("output_subdir"):
                variant_report_subdir = variant_report_subdir / str(variant["output_subdir"])
            output_dir = report_root / variant_report_subdir
            log_dir = output_dir / "logs"
            wandb_group = spec["wandb_group"] if variant_id == "default" else f"{spec['wandb_group']}/{variant_id}"
            display_baseline = model_variant_display_name(baseline, variant_id)
            for category in categories:
                summary_path = output_dir / f"{category}_{output_suffix}.json"
                log_path = log_dir / f"{category}_{output_suffix}.log.txt"
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
                    "--output",
                    str(output_dir),
                    "--log-dir",
                    str(log_dir),
                    *variant["extra_args"],
                ]
                if normalized_severities:
                    runner_cmd.extend(["--severities", *normalized_severities])
                if spec["use_wandb"]:
                    runner_cmd.extend(
                        [
                            "--use-wandb",
                            "--wandb-project",
                            wandb_project,
                            "--wandb-group",
                            wandb_group,
                            "--wandb-mode",
                            wandb_mode,
                        ]
                    )
                    if spec["wandb_log_images"]:
                        runner_cmd.extend(["--wandb-log-images", "--wandb-max-images", str(spec["wandb_max_images"])])
                configs.append(
                    {
                        "baseline": baseline,
                        "display_baseline": display_baseline,
                        "category": category,
                        "device": spec["device"],
                        "data_root": str(spec["data_root"]),
                        "manifest_paths": list(manifest_paths),
                        "manifest_names": list(manifest_names),
                        "selected_severities": list(normalized_severities),
                        "summary_path": summary_path,
                        "log_path": log_path,
                        "runner_cmd": runner_cmd,
                        "use_wandb": spec["use_wandb"],
                        "runner_name": spec["runner_name"],
                        "runner_path": spec["runner_path"],
                        "runner_inputs": spec["runner_inputs"],
                        "runner_outputs": spec["runner_outputs"],
                        "wandb_group": wandb_group,
                        "wandb_log_images": spec["wandb_log_images"],
                        "wandb_max_images": spec["wandb_max_images"],
                        "model_variant_id": variant_id,
                        "model_variant_label": variant.get("label", variant_id),
                        "model_sweep_values": dict(variant.get("sweep_values", {})),
                        "model_config": dict(variant.get("config", {})),
                        "extra_args": list(variant.get("extra_args", [])),
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
                    "selected_severities": ", ".join(
                        requested_run_configs[0].get("selected_severities", [])
                    )
                    if requested_run_configs and requested_run_configs[0].get("selected_severities")
                    else "all",
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
                    "model_config": format_model_config(specs[baseline].get("model_config", {})),
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
        label = f"{config.get('display_baseline', config['baseline'])} / {config['category']}"
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
                "display_baseline": config.get("display_baseline", config["baseline"]),
                "model_variant": config.get("model_variant_id", "default"),
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
