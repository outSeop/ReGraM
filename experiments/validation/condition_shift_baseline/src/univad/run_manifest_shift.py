"""Run UniVAD condition-shift evaluation from a manifest."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:  # pragma: no cover - supports script execution and package imports.
    from .transformers_runtime import disable_transformers_tensorflow_backend
except ImportError:  # pragma: no cover
    from transformers_runtime import disable_transformers_tensorflow_backend


disable_transformers_tensorflow_backend()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


UNIVAD_IMPORT_DEPENDENCIES = {
    "ftfy": "ftfy",
    "regex": "regex",
    "tqdm": "tqdm",
}


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
    attach_heatmap_artifacts,
    build_clean_metric_snapshot,
    build_clean_sample_rows,
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
    disable_transformers_tensorflow_backend()
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


def ensure_univad_import_dependencies() -> list[str]:
    missing_specs = []
    for module_name, package_spec in UNIVAD_IMPORT_DEPENDENCIES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_specs.append(package_spec)
    if not missing_specs:
        return []
    print(f"install missing UniVAD import dependencies: {', '.join(missing_specs)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", *missing_specs],
        check=True,
    )
    return missing_specs


def patch_univad_optional_pydensecrf_file(univad_root: Path) -> bool:
    crf_path = univad_root / "utils" / "crf.py"
    if not crf_path.exists():
        return False

    text = crf_path.read_text(encoding="utf-8")
    if "dcrf = None" in text and "if dcrf is None or utils is None:" in text:
        return False

    import_target = "import pydensecrf.densecrf as dcrf\nimport pydensecrf.utils as utils\n"
    import_replacement = (
        "try:\n"
        "    import pydensecrf.densecrf as dcrf\n"
        "    import pydensecrf.utils as utils\n"
        "except ImportError:\n"
        "    dcrf = None\n"
        "    utils = None\n"
    )
    function_target = (
        "def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):\n"
        "    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]\n"
    )
    function_replacement = (
        "def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):\n"
        "    if dcrf is None or utils is None:\n"
        "        output_logits = F.interpolate(\n"
        "            output_logits.float().unsqueeze(0),\n"
        "            size=image_tensor.shape[-2:],\n"
        "            mode=\"bilinear\",\n"
        "            align_corners=False,\n"
        "        ).squeeze()\n"
        "        return F.softmax(output_logits, dim=0).cpu().numpy()\n"
        "\n"
        "    image = np.array(VF.to_pil_image(unnorm(image_tensor.float())))[:, :, ::-1]\n"
    )
    if import_target not in text or function_target not in text:
        return False
    text = text.replace(import_target, import_replacement)
    text = text.replace(function_target, function_replacement)
    crf_path.write_text(text, encoding="utf-8")
    print(f"patch UniVAD optional pydensecrf fallback: {crf_path}")
    return True


def patch_univad_reseg_loop_guard(univad_root: Path) -> bool:
    univad_path = univad_root / "UniVAD.py"
    if not univad_path.exists():
        return False

    text = univad_path.read_text(encoding="utf-8")
    if "REGRAM_UNIVAD_MAX_RESEG_ATTEMPTS" in text:
        return False

    target = (
        "                while part_num not in part_num_right:\n"
        "                    kmeans = KMeans(init=\"k-means++\", n_clusters=n_cluster)\n"
    )
    replacement = (
        "                regram_max_reseg_attempts = int(os.environ.get(\"REGRAM_UNIVAD_MAX_RESEG_ATTEMPTS\", \"3\"))\n"
        "                regram_reseg_attempt = 0\n"
        "                while part_num not in part_num_right:\n"
        "                    regram_reseg_attempt += 1\n"
        "                    if regram_reseg_attempt > regram_max_reseg_attempts:\n"
        "                        print(\n"
        "                            \"warning: UniVAD re-seg loop did not reach expected part count; \"\n"
        "                            f\"part_num={part_num}, expected={part_num_right}, \"\n"
        "                            f\"attempts={regram_max_reseg_attempts}. Continuing with last heat masks.\",\n"
        "                            flush=True,\n"
        "                        )\n"
        "                        break\n"
        "                    print(\n"
        "                        \"UniVAD re-seg attempt \"\n"
        "                        f\"{regram_reseg_attempt}/{regram_max_reseg_attempts}: \"\n"
        "                        f\"part_num={part_num}, expected={part_num_right}\",\n"
        "                        flush=True,\n"
        "                    )\n"
        "                    kmeans = KMeans(init=\"k-means++\", n_clusters=n_cluster)\n"
    )
    if target not in text:
        return False
    univad_path.write_text(text.replace(target, replacement), encoding="utf-8")
    print(f"patch UniVAD re-seg loop guard: {univad_path}")
    return True


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
def timeout_guard(seconds: int, *, phase_name: str):
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handle_timeout(signum, frame):  # noqa: ANN001, ARG001
        raise TimeoutError(f"{phase_name} exceeded timeout_sec={seconds}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def collect_heat_mask_debug_stats(univad_root: Path, category: str) -> str:
    heat_root = univad_root / "heat_masks"
    if not heat_root.exists():
        return f"heat_masks_root_missing={heat_root}"
    category_dirs = sorted(path for path in heat_root.glob(f"{category}*") if path.is_dir())
    png_count = sum(len(list(path.rglob("*.png"))) for path in category_dirs)
    dir_names = ",".join(path.name for path in category_dirs) if category_dirs else "-"
    return f"heat_masks_root={heat_root} | category_dirs={dir_names} | png_count={png_count}"


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


def predict_image_once(
    model,
    image: Image.Image,
    image_path: str,
    transform,
    device: torch.device,
    *,
    image_size: int,
    use_amp: bool,
    clear_cache: bool,
    return_map: bool = False,
) -> tuple[float, np.ndarray | None]:
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
        score = float(pred_score.item() if torch.is_tensor(pred_score) else pred_score)
        pred_map = None
        if return_map:
            pred_mask = pred.get("pred_mask")
            if torch.is_tensor(pred_mask):
                pred_map = pred_mask.detach().float().cpu().numpy()
            elif pred_mask is not None:
                pred_map = np.asarray(pred_mask, dtype=np.float32)
        return score, pred_map
    finally:
        del batched
        del image_pil_list
        with contextlib.suppress(UnboundLocalError):
            del pred
        clear_cuda_cache(device, clear_cache)


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
    score, _ = predict_image_once(
        model,
        image,
        image_path,
        transform,
        device,
        image_size=image_size,
        use_amp=use_amp,
        clear_cache=clear_cache,
        return_map=False,
    )
    return score


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


def score_image_with_map(
    model,
    image: Image.Image,
    image_path: str,
    transform,
    device: torch.device,
    *,
    image_size: int,
    use_amp: bool,
    clear_cache: bool,
) -> tuple[float, np.ndarray | None]:
    try:
        return predict_image_once(
            model,
            image,
            image_path,
            transform,
            device,
            image_size=image_size,
            use_amp=use_amp,
            clear_cache=clear_cache,
            return_map=True,
        )
    except RuntimeError as exc:
        if not use_amp or not amp_unsupported_error(exc):
            raise
        clear_cuda_cache(device, clear_cache)
        return predict_image_once(
            model,
            image,
            image_path,
            transform,
            device,
            image_size=image_size,
            use_amp=False,
            clear_cache=clear_cache,
            return_map=True,
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
    parser.add_argument("--export-heatmaps", action="store_true")
    parser.add_argument("--heatmap-max-images", type=int, default=2)
    parser.add_argument(
        "--setup-timeout-sec",
        type=int,
        default=1800,
        help="Timeout guard for model.setup phase. Set 0 to disable.",
    )
    parser.add_argument(
        "--setup-re-seg",
        choices=["auto", "true", "false"],
        default="auto",
        help="Force UniVAD setup re-seg behavior if model.setup supports re_seg.",
    )
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
    installed_import_dependencies = ensure_univad_import_dependencies()
    patched_optional_pydensecrf = patch_univad_optional_pydensecrf_file(univad_root)
    patched_reseg_loop_guard = patch_univad_reseg_loop_guard(univad_root)
    from UniVAD import UniVAD  # noqa: WPS433

    patched_dense_crf = patch_univad_dense_crf_float32()
    finish_phase(
        phase_logs,
        "imports",
        phase_started_at,
        extra=(
            f"patched_dense_crf={patched_dense_crf} | "
            f"patched_optional_pydensecrf={patched_optional_pydensecrf} | "
            f"patched_reseg_loop_guard={patched_reseg_loop_guard} | "
            f"installed_import_dependencies={installed_import_dependencies or '-'}"
        ),
    )

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

    output_paths = prepare_output_paths(
        output_root=args.output,
        log_dir=args.log_dir,
        category=args.category,
        output_suffix=run_spec.output_suffix,
    )
    explainability_dir = output_paths.output_dir / "explainability" / args.category / run_spec.output_suffix

    transform = build_transform(args.image_size)
    train_good_dir = data_root / args.category / "train" / "good"
    clean_good_dir = data_root / args.category / "test" / "good"
    logical_dir = data_root / args.category / "test" / "logical_anomalies"
    structural_dir = data_root / args.category / "test" / "structural_anomalies"
    clean_paths = sorted(clean_good_dir.glob("*.png"))
    logical_anomaly_paths = sorted(logical_dir.glob("*.png"))
    structural_anomaly_paths = sorted(structural_dir.glob("*.png"))
    anomaly_paths = logical_anomaly_paths + structural_anomaly_paths
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
        setup_payload = {
            "few_shot_samples": reference_tensor,
            "dataset_category": args.category,
            "image_path": reference_paths,
        }
        setup_kwargs: dict[str, object] = {}
        setup_signature = inspect.signature(model.setup)
        if "re_seg" in setup_signature.parameters and args.setup_re_seg != "auto":
            setup_kwargs["re_seg"] = args.setup_re_seg == "true"
        log_phase(
            phase_logs,
            "setup_few_shot_begin",
            (
                f"k_shot={args.k_shot} | round={args.round} | "
                f"setup_re_seg={setup_kwargs.get('re_seg', 'auto')} | "
                f"{collect_heat_mask_debug_stats(univad_root, args.category)}"
            ),
        )
        try:
            with timeout_guard(args.setup_timeout_sec, phase_name="setup_few_shot"):
                with cuda_autocast_context(device, effective_amp):
                    model.setup(setup_payload, **setup_kwargs)
        except TimeoutError as exc:
            debug_stats = collect_heat_mask_debug_stats(univad_root, args.category)
            raise RuntimeError(
                "UniVAD model.setup timed out. "
                f"category={args.category} | {debug_stats} | "
                "Try --setup-re-seg false or increase --setup-timeout-sec. "
                "If external UniVAD re-seg loop is stuck, inspect heat_masks and filter_bg_noise."
            ) from exc
        clear_cuda_cache(device, clear_cache)
        finish_phase(
            phase_logs,
            "setup_few_shot",
            phase_started_at,
            extra=(
                f"k_shot={args.k_shot} | round={args.round} | "
                f"setup_re_seg={setup_kwargs.get('re_seg', 'auto')} | "
                f"{collect_heat_mask_debug_stats(univad_root, args.category)}"
            ),
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
        logical_anomaly_scores: list[float] = []
        for path in logical_anomaly_paths:
            with Image.open(path) as image_obj:
                image = image_obj.convert("RGB")
            logical_anomaly_scores.append(
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
            "score_clean_logical_anomaly",
            phase_started_at,
            extra=f"samples={len(logical_anomaly_paths)}",
        )

        phase_started_at = time.perf_counter()
        structural_anomaly_scores: list[float] = []
        for path in structural_anomaly_paths:
            with Image.open(path) as image_obj:
                image = image_obj.convert("RGB")
            structural_anomaly_scores.append(
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
            "score_clean_structural_anomaly",
            phase_started_at,
            extra=f"samples={len(structural_anomaly_paths)}",
        )
        anomaly_scores = logical_anomaly_scores + structural_anomaly_scores

        clean_threshold = max(clean_scores) if clean_scores else 0.0
        results = build_results_scaffold(
            updated_at=now_kst_string(),
            category=args.category,
            run_spec=run_spec,
            clean_good=summarize_scores(clean_scores, clean_threshold),
            clean_anomaly=summarize_scores(anomaly_scores, clean_threshold),
            clean_image_auroc=compute_image_auroc(clean_scores, anomaly_scores),
            clean_logical_anomaly=summarize_scores(logical_anomaly_scores, clean_threshold),
            clean_structural_anomaly=summarize_scores(structural_anomaly_scores, clean_threshold),
            clean_logical_image_auroc=compute_image_auroc(clean_scores, logical_anomaly_scores),
            clean_structural_image_auroc=compute_image_auroc(clean_scores, structural_anomaly_scores),
        )
        results["clean_score_distributions"] = {
            "clean_normal_scores": clean_scores,
            "clean_anomaly_scores": anomaly_scores,
            "clean_logical_anomaly_scores": logical_anomaly_scores,
            "clean_structural_anomaly_scores": structural_anomaly_scores,
        }
        if args.export_heatmaps:
            clean_reference_maps: dict[str, list[dict[str, object]]] = {}
            clean_reference_specs = [
                ("clean_normal", clean_paths, clean_scores),
                ("logical_anomaly", logical_anomaly_paths, logical_anomaly_scores),
                ("structural_anomaly", structural_anomaly_paths, structural_anomaly_scores),
            ]
            for sample_type, image_paths_for_split, scores_for_split in clean_reference_specs:
                sample_rows = build_clean_sample_rows(
                    image_paths=image_paths_for_split,
                    scores=scores_for_split,
                    clean_good_mean=results["clean_good"]["mean"],
                    sample_type=sample_type,
                    max_items=args.heatmap_max_images,
                )
                heatmaps_by_index: dict[int, np.ndarray] = {}
                for row in sample_rows:
                    source_index = int(row["source_index"])
                    image_path = image_paths_for_split[source_index]
                    with Image.open(image_path) as image_obj:
                        image = image_obj.convert("RGB")
                    _, pred_map = score_image_with_map(
                        model,
                        image,
                        str(image_path),
                        transform,
                        device,
                        image_size=args.image_size,
                        use_amp=effective_amp,
                        clear_cache=clear_cache,
                    )
                    if pred_map is not None:
                        heatmaps_by_index[source_index] = pred_map
                clean_reference_maps[sample_type] = attach_heatmap_artifacts(
                    sample_rows=sample_rows,
                    heatmaps_by_index=heatmaps_by_index,
                    artifact_dir=explainability_dir / "clean_reference" / sample_type,
                    artifact_prefix=f"univad_{sample_type}",
                    apply_shift=False,
                )
            results["clean_reference_maps"] = clean_reference_maps

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
                summary["shifted_image_auroc_vs_logical_anomaly"] = compute_image_auroc(
                    shifted_scores,
                    logical_anomaly_scores,
                )
                summary["shifted_image_auroc_vs_structural_anomaly"] = compute_image_auroc(
                    shifted_scores,
                    structural_anomaly_scores,
                )
                summary["image_auroc_drop_from_clean"] = (
                    results["clean_image_auroc"] - summary["shifted_image_auroc"]
                )
                summary["image_auroc_drop_from_clean_logical"] = (
                    results["clean_logical_image_auroc"]
                    - summary["shifted_image_auroc_vs_logical_anomaly"]
                )
                summary["image_auroc_drop_from_clean_structural"] = (
                    results["clean_structural_image_auroc"]
                    - summary["shifted_image_auroc_vs_structural_anomaly"]
                )
                summary["shifted_normal_scores"] = shifted_scores
                summary["sample_rows"] = build_shift_sample_rows(
                    entries=entries,
                    scores=shifted_scores,
                    clean_good_mean=results["clean_good"]["mean"],
                    max_items=max(args.heatmap_max_images, args.wandb_max_images),
                )
                if args.export_heatmaps:
                    heatmaps_by_index: dict[int, np.ndarray] = {}
                    for row in summary["sample_rows"]:
                        source_index = int(row["source_index"])
                        entry = entries[source_index]
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
                        _, pred_map = score_image_with_map(
                            model,
                            image,
                            str(model_image_path),
                            transform,
                            device,
                            image_size=args.image_size,
                            use_amp=effective_amp,
                            clear_cache=clear_cache,
                        )
                        if pred_map is not None:
                            heatmaps_by_index[source_index] = pred_map
                    summary["sample_rows"] = attach_heatmap_artifacts(
                        sample_rows=summary["sample_rows"],
                        heatmaps_by_index=heatmaps_by_index,
                        artifact_dir=explainability_dir / "shifted" / aug_type / severity,
                        artifact_prefix=f"univad_{aug_type}_{severity}",
                        apply_shift=True,
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
            "setup_timeout_sec": args.setup_timeout_sec,
            "setup_re_seg": args.setup_re_seg,
            "export_heatmaps": args.export_heatmaps,
            "heatmap_max_images": args.heatmap_max_images,
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
