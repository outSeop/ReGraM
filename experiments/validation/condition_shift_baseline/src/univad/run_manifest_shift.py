"""Run UniVAD condition-shift evaluation from a manifest."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def ensure_numpy_1_runtime() -> None:
    target_version = "1.26.4"
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = "-"
    major_version = numpy_version.split(".", maxsplit=1)[0]
    if major_version == "1":
        return
    print(f"install numpy=={target_version} before torch import; current numpy={numpy_version}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", f"numpy=={target_version}"],
        check=True,
    )


ensure_numpy_1_runtime()

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from augmentation_runtime import apply_augmentation
from contracts import write_log
from external_loader import ensure_external_on_path
from manifest_shift_common import (
    DEFAULT_GALLERY_SAMPLE_LIMIT,
    build_clean_metric_snapshot,
    build_common_run_config,
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
from manifest_paths import resolve_manifest_image_path
from preview_utils import build_preview_images
from repo_paths import REPO_ROOT, finish_phase, log_phase, now_kst_string


@contextlib.contextmanager
def univad_runtime_context(univad_root: Path):
    previous_cwd = Path.cwd()
    os.environ.setdefault("TORCH_HOME", str(univad_root / "pretrained_ckpts" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.chdir(univad_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def ensure_univad_import_paths(univad_root: Path) -> None:
    groundingdino_root = univad_root / "models" / "GroundingDINO"
    for path in (univad_root, groundingdino_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class ResizeToTensorNoNumpy:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)
        image_bytes = resized.convert("RGB").tobytes()
        tensor = torch.frombuffer(image_bytes, dtype=torch.uint8).clone()
        tensor = tensor.view(self.image_size, self.image_size, 3)
        return tensor.permute(2, 0, 1).contiguous().float().div(255.0)


def build_transform(image_size: int):
    return ResizeToTensorNoNumpy(image_size)


def pil_to_univad_inputs(image: Image.Image, *, image_size: int, transform, device: torch.device):
    tensor = transform(image)
    batched = tensor.unsqueeze(0).to(device)
    resized = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    image_pil_list = [np.asarray(resized.convert("RGB"), dtype=np.uint8)]
    return batched, image_pil_list


def clear_cuda_cache(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.empty_cache()


@contextlib.contextmanager
def low_memory_univad_model_load(enabled: bool):
    original_load = torch.hub.load
    univad_module = sys.modules.get("UniVAD")
    open_clip_module = getattr(univad_module, "open_clip", None)
    original_create_model_and_transforms = (
        getattr(open_clip_module, "create_model_and_transforms", None)
        if open_clip_module is not None
        else None
    )

    if not enabled:
        yield
        return

    def patched_load(repo_or_dir, model, *args, **kwargs):
        loaded_model = original_load(repo_or_dir, model, *args, **kwargs)
        if str(model) == "dinov2_vitg14":
            print("load DINOv2 ViT-G/14 backbone in fp16 for low-memory CUDA runtime")
            return loaded_model.half()
        return loaded_model

    def patched_create_model_and_transforms(*args, **kwargs):
        model, preprocess_train, preprocess_val = original_create_model_and_transforms(*args, **kwargs)
        print("load CLIP backbone in fp16 for low-memory CUDA runtime")
        return model.half(), preprocess_train, preprocess_val

    torch.hub.load = patched_load
    if open_clip_module is not None and original_create_model_and_transforms is not None:
        open_clip_module.create_model_and_transforms = patched_create_model_and_transforms
    try:
        yield
    finally:
        torch.hub.load = original_load
        if open_clip_module is not None and original_create_model_and_transforms is not None:
            open_clip_module.create_model_and_transforms = original_create_model_and_transforms


def cuda_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def patch_univad_dense_crf_float32() -> bool:
    univad_module = sys.modules.get("UniVAD")
    original_dense_crf = getattr(univad_module, "dense_crf", None) if univad_module is not None else None
    if original_dense_crf is None or getattr(original_dense_crf, "_regram_float32_patch", False):
        return False

    def dense_crf_float32(image_tensor, output_logits):
        if torch.is_tensor(image_tensor):
            image_tensor = image_tensor.float()
        if torch.is_tensor(output_logits):
            output_logits = output_logits.float()
        return original_dense_crf(image_tensor, output_logits)

    dense_crf_float32._regram_float32_patch = True
    univad_module.dense_crf = dense_crf_float32
    print("patch UniVAD dense_crf inputs to fp32")
    return True


def amp_unsupported_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "not implemented for 'Half'" in message or "not implemented for Half" in message


def score_image_once(
    model,
    image: Image.Image,
    image_path: str,
    transform,
    device: torch.device,
    *,
    image_size: int,
    use_amp: bool,
    clear_cache: bool,
) -> float:
    batched, image_pil_list = pil_to_univad_inputs(
        image,
        image_size=image_size,
        transform=transform,
        device=device,
    )
    amp_context = cuda_autocast_context(device, use_amp)
    try:
        with torch.inference_mode(), amp_context:
            pred = model(batched, image_path, image_pil_list)
        pred_score = pred["pred_score"]
        return float(pred_score.item() if torch.is_tensor(pred_score) else pred_score)
    finally:
        del batched
        del image_pil_list
        with contextlib.suppress(UnboundLocalError):
            del pred
        clear_cuda_cache(device, clear_cache)


def score_image(
    model,
    image: Image.Image,
    image_path: str,
    transform,
    device: torch.device,
    *,
    image_size: int,
    use_amp: bool,
    clear_cache: bool,
) -> float:
    try:
        return score_image_once(
            model,
            image,
            image_path,
            transform,
            device,
            image_size=image_size,
            use_amp=use_amp,
            clear_cache=clear_cache,
        )
    except RuntimeError as exc:
        if not use_amp or not amp_unsupported_error(exc):
            raise
        clear_cuda_cache(device, clear_cache)
        return score_image_once(
            model,
            image,
            image_path,
            transform,
            device,
            image_size=image_size,
            use_amp=False,
            clear_cache=clear_cache,
        )


def select_reference_paths(train_good_dir: Path, *, k_shot: int, round_idx: int) -> list[Path]:
    paths = sorted(train_good_dir.glob("*.png"))
    if len(paths) < round_idx + k_shot:
        raise RuntimeError(
            f"Not enough train/good images in {train_good_dir} for k_shot={k_shot}, round={round_idx}. "
            f"Available={len(paths)}."
        )
    return paths[round_idx : round_idx + k_shot]


def load_reference_tensors(
    train_good_dir: Path, *, k_shot: int, round_idx: int, transform, device
) -> tuple[torch.Tensor, list[str]]:
    selected = select_reference_paths(train_good_dir, k_shot=k_shot, round_idx=round_idx)
    tensors = torch.cat(
        [transform(Image.open(path).convert("RGB")).unsqueeze(0) for path in selected],
        dim=0,
    ).to(device)
    return tensors, [str(path) for path in selected]


def compute_image_auroc(normal_scores: list[float], anomaly_scores: list[float]) -> float:
    if not normal_scores or not anomaly_scores:
        return 0.0
    labels = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    scores = normal_scores + anomaly_scores
    return float(roc_auc_score(labels, scores) * 100.0)


def univad_data_path_for_image(image_path: Path, *, data_root: Path, category: str) -> Path:
    category_root = data_root / category
    with contextlib.suppress(ValueError):
        image_path.relative_to(category_root)
        return image_path
    parts = list(image_path.parts)
    if category in parts:
        category_index = len(parts) - 1 - parts[::-1].index(category)
        rel_path = Path(*parts[category_index + 1 :])
        return category_root / rel_path
    return image_path


def expected_mask_path_for_image(image_path: Path, *, data_root: Path, mask_root: Path, category: str) -> Path:
    data_image_path = univad_data_path_for_image(image_path, data_root=data_root, category=category)
    rel_path = data_image_path.relative_to(data_root / category)
    return mask_root / category / rel_path.with_suffix("") / "grounding_mask.png"


def collect_manifest_source_paths(run_spec) -> list[Path]:
    paths: list[Path] = []
    for severity_groups in run_spec.grouped_entries.values():
        for entries in severity_groups.values():
            paths.extend(resolve_manifest_image_path(entry) for entry in entries)
    return paths


def assert_required_grounding_masks_exist(
    *,
    data_root: Path,
    mask_root: Path,
    category: str,
    image_paths: list[Path],
) -> None:
    expected_paths: dict[Path, Path] = {}
    for image_path in image_paths:
        try:
            expected_paths[image_path] = expected_mask_path_for_image(
                image_path,
                data_root=data_root,
                mask_root=mask_root,
                category=category,
            )
        except ValueError:
            continue
    missing = [mask_path for mask_path in expected_paths.values() if not mask_path.exists()]
    if not missing:
        return
    preview = "\n".join(str(path) for path in missing[:10])
    more = f"\n... +{len(missing) - 10} more" if len(missing) > 10 else ""
    raise RuntimeError(
        "UniVAD grounding masks are incomplete for this run. "
        "Run the notebook setup/readiness cell to generate masks for all category images before scoring.\n"
        f"missing_count={len(missing)}\n{preview}{more}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--manifest", nargs="+")
    parser.add_argument("--augmentation-type")
    parser.add_argument("--augmentation-types", nargs="+")
    parser.add_argument("--severities", nargs="+")
    parser.add_argument("--input-root", default="data/query_normal_clean")
    parser.add_argument("--data-root", default="data/row/mvtec_loco_anomaly_detection")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast during UniVAD scoring.")
    parser.add_argument(
        "--disable-cuda-empty-cache",
        action="store_true",
        help="Do not call torch.cuda.empty_cache() between UniVAD scoring calls.",
    )
    parser.add_argument(
        "--disable-low-memory-backbone",
        action="store_true",
        help="Keep the DINOv2 ViT-G/14 backbone in fp32 instead of fp16.",
    )
    parser.add_argument(
        "--output",
        default="experiments/validation/condition_shift_baseline/reports/univad_manifest_shift",
    )
    parser.add_argument(
        "--log-dir",
        default="experiments/validation/condition_shift_baseline/reports/univad_manifest_shift/logs",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="regram-condition-shift")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="univad")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-images", action="store_true")
    parser.add_argument("--wandb-max-images", type=int, default=2)
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        raise RuntimeError(
            "UniVAD manifest-shift requires CUDA. Re-run on a GPU runtime with --device cuda."
        )

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
    univad_root = ensure_external_on_path("univad")
    ensure_univad_import_paths(univad_root)
    from UniVAD import UniVAD  # noqa: WPS433

    patched_dense_crf = patch_univad_dense_crf_float32()
    finish_phase(phase_logs, "imports", phase_started_at, extra=f"patched_dense_crf={patched_dense_crf}")

    device = torch.device(args.device)
    clear_cache = not args.disable_cuda_empty_cache
    low_memory_backbone = device.type == "cuda" and not args.disable_low_memory_backbone
    effective_amp = args.amp or low_memory_backbone
    data_root = resolve_repo_path(args.data_root)
    input_root = resolve_repo_path(args.input_root)

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

    transform = build_transform(args.image_size)
    train_good_dir = data_root / args.category / "train" / "good"
    clean_good_dir = data_root / args.category / "test" / "good"
    logical_dir = data_root / args.category / "test" / "logical_anomalies"
    structural_dir = data_root / args.category / "test" / "structural_anomalies"
    clean_paths = sorted(clean_good_dir.glob("*.png"))
    anomaly_paths = sorted(logical_dir.glob("*.png")) + sorted(structural_dir.glob("*.png"))
    reference_paths_for_masks = select_reference_paths(
        train_good_dir,
        k_shot=args.k_shot,
        round_idx=args.round,
    )
    required_image_paths = list(
        dict.fromkeys(
            [
                *reference_paths_for_masks,
                *clean_paths,
                *anomaly_paths,
                *collect_manifest_source_paths(run_spec),
            ]
        )
    )
    mask_root = univad_root / "masks" / "mvtec_loco_caption"

    phase_started_at = time.perf_counter()
    assert_required_grounding_masks_exist(
        data_root=data_root,
        mask_root=mask_root,
        category=args.category,
        image_paths=required_image_paths,
    )
    finish_phase(
        phase_logs,
        "mask_preflight",
        phase_started_at,
        extra=f"required_images={len(required_image_paths)} | mask_root={mask_root}",
    )

    preview_images: dict[str, list[dict[str, object]]] = {}

    with univad_runtime_context(univad_root):
        phase_started_at = time.perf_counter()
        clear_cuda_cache(device, clear_cache)
        constructor_kwargs = {"image_size": args.image_size}
        constructor_accepts_device = "device" in inspect.signature(UniVAD.__init__).parameters
        if constructor_accepts_device:
            constructor_kwargs["device"] = device
        with low_memory_univad_model_load(low_memory_backbone), cuda_autocast_context(device, effective_amp):
            model = UniVAD(**constructor_kwargs)
        if not constructor_accepts_device:
            model = model.to(device)
        model.eval()
        finish_phase(
            phase_logs,
            "model_build",
            phase_started_at,
            extra=f"low_memory_backbone={low_memory_backbone} | amp={effective_amp}",
        )

        phase_started_at = time.perf_counter()
        reference_tensor, reference_paths = load_reference_tensors(
            train_good_dir,
            k_shot=args.k_shot,
            round_idx=args.round,
            transform=transform,
            device=device,
        )
        with cuda_autocast_context(device, effective_amp):
            model.setup(
                {
                    "few_shot_samples": reference_tensor,
                    "dataset_category": args.category,
                    "image_path": reference_paths,
                }
            )
        clear_cuda_cache(device, clear_cache)
        finish_phase(
            phase_logs,
            "setup_few_shot",
            phase_started_at,
            extra=f"k_shot={args.k_shot} | round={args.round}",
        )

        phase_started_at = time.perf_counter()
        clean_scores: list[float] = []
        for path in clean_paths:
            with Image.open(path) as image_obj:
                image = image_obj.convert("RGB")
            clean_scores.append(
                score_image(
                    model,
                    image,
                    str(path),
                    transform,
                    device,
                    image_size=args.image_size,
                    use_amp=effective_amp,
                    clear_cache=clear_cache,
                )
            )
        finish_phase(
            phase_logs,
            "score_clean_good",
            phase_started_at,
            extra=f"samples={len(clean_paths)}",
        )

        phase_started_at = time.perf_counter()
        anomaly_scores: list[float] = []
        for path in anomaly_paths:
            with Image.open(path) as image_obj:
                image = image_obj.convert("RGB")
            anomaly_scores.append(
                score_image(
                    model,
                    image,
                    str(path),
                    transform,
                    device,
                    image_size=args.image_size,
                    use_amp=effective_amp,
                    clear_cache=clear_cache,
                )
            )
        finish_phase(
            phase_logs,
            "score_clean_anomaly",
            phase_started_at,
            extra=f"samples={len(anomaly_paths)}",
        )

        clean_threshold = max(clean_scores) if clean_scores else 0.0
        results = build_results_scaffold(
            updated_at=now_kst_string(),
            category=args.category,
            run_spec=run_spec,
            clean_good=summarize_scores(clean_scores, clean_threshold),
            clean_anomaly=summarize_scores(anomaly_scores, clean_threshold),
            clean_image_auroc=compute_image_auroc(clean_scores, anomaly_scores),
        )
        results["clean_score_distributions"] = {
            "clean_normal_scores": clean_scores,
            "clean_anomaly_scores": anomaly_scores,
        }

        for aug_type, severity_groups in sorted(run_spec.grouped_entries.items()):
            for severity, entries in sorted(severity_groups.items()):
                phase_started_at = time.perf_counter()
                shifted_scores: list[float] = []
                for entry in entries:
                    image_path = resolve_manifest_image_path(entry)
                    with Image.open(image_path) as image_obj:
                        image = image_obj.convert("RGB")
                    image = apply_augmentation(
                        image,
                        augmentation_type=entry["augmentation_type"],
                        severity=entry["severity"],
                        seed=entry["seed"],
                        params=entry["params"],
                    )
                    model_image_path = univad_data_path_for_image(
                        image_path,
                        data_root=data_root,
                        category=args.category,
                    )
                    shifted_scores.append(
                        score_image(
                            model,
                            image,
                            str(model_image_path),
                            transform,
                            device,
                            image_size=args.image_size,
                            use_amp=effective_amp,
                            clear_cache=clear_cache,
                        )
                    )

                summary = summarize_scores(shifted_scores, clean_threshold)
                summary["shifted_normal_fpr"] = summary["fpr_over_clean_max"]
                summary["mean_score_shift"] = summary["mean"] - results["clean_good"]["mean"]
                summary["median_score_shift"] = (
                    summary["median"] - results["clean_good"]["median"]
                )
                summary["shifted_image_auroc"] = compute_image_auroc(
                    shifted_scores,
                    anomaly_scores,
                )
                summary["image_auroc_vs_clean_anomaly"] = summary["shifted_image_auroc"]
                summary["image_auroc_drop_from_clean"] = (
                    results["clean_image_auroc"] - summary["shifted_image_auroc"]
                )
                summary["shifted_normal_scores"] = shifted_scores
                summary["sample_rows"] = build_shift_sample_rows(
                    entries=entries,
                    scores=shifted_scores,
                    clean_good_mean=results["clean_good"]["mean"],
                    max_items=max(DEFAULT_GALLERY_SAMPLE_LIMIT, args.wandb_max_images),
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
                    extra=f"samples={len(entries)}",
                )

    output_paths = prepare_output_paths(
        output_root=args.output,
        log_dir=args.log_dir,
        category=args.category,
        output_suffix=run_spec.output_suffix,
    )
    common_run_config = build_common_run_config(
        run_spec=run_spec,
        input_root=args.input_root,
        data_root=args.data_root,
        device=args.device,
        wandb_log_images=args.wandb_log_images,
        wandb_max_images=args.wandb_max_images,
        extra_config={
            "image_size": args.image_size,
            "k_shot": args.k_shot,
            "round": args.round,
            "amp": effective_amp,
            "requested_amp": args.amp,
            "low_memory_backbone": low_memory_backbone,
            "patched_dense_crf": patched_dense_crf,
            "cuda_empty_cache": clear_cache,
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
            baseline="UniVAD",
            dataset="mvtec_loco",
            class_name=args.category,
            eval_type="manifest_shift",
            run_name=build_run_name("univad", args.category, run_spec.output_suffix),
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
        baseline="UniVAD",
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
            "univad_root": univad_root,
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
            "runner=run_manifest_shift.py",
            "baseline=UniVAD",
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
