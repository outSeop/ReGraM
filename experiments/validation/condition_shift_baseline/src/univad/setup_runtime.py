"""Colab/runtime setup helpers for UniVAD notebook execution."""

from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from notebook_orchestration import ensure_importable_path, module_available


UNIVAD_RUNTIME_DEPENDENCIES = [
    "addict",
    "yapf",
    "pycocotools",
    "supervision",
    "timm",
    "torchmetrics",
    "wget",
    "ftfy",
    "regex",
    "tqdm",
]
UNIVAD_RUNTIME_DEPENDENCY_SPECS = [
    "addict==2.4.0",
    "yapf==0.40.2",
    "pycocotools==2.0.8",
    "supervision",
    "timm==1.0.9",
    "torchmetrics",
    "wget",
    "ftfy",
    "regex",
    "tqdm",
]
UNIVAD_NUMPY_VERSION = "1.26.4"
UNIVAD_OPENCV_VERSION = "4.11.0.86"
UNIVAD_TRANSFORMERS_VERSION = "4.45.2"
UNIVAD_TORCH_VERSION = "2.2.0"
UNIVAD_TORCHVISION_VERSION = "0.17.0"
UNIVAD_TORCHAUDIO_VERSION = "2.2.0"


def default_univad_setup_flags() -> dict[str, Any]:
    return {
        "download_missing_univad_checkpoints": True,
        "auto_prepare_univad_caption_dataset": True,
        "auto_prepare_univad_grounding_masks": True,
        "auto_fix_univad_torch_stack": True,
        "auto_fix_univad_transformers_stack": True,
        "auto_fix_univad_runtime_deps": True,
        "restore_univad_grounding_masks_from_drive": True,
        "backup_univad_grounding_masks_to_drive": True,
        "auto_mount_google_drive_for_univad_masks": True,
        "univad_mask_drive_backup_root": "/content/drive/MyDrive/ReGraM/univad_masks/mvtec_loco_caption",
    }


def collect_missing_checkpoint_files(spec: dict[str, Any]) -> list[str]:
    checkpoint_root = spec.get("checkpoint_root")
    if checkpoint_root is None:
        return []
    return [
        filename
        for filename in spec.get("required_checkpoint_files", [])
        if not (checkpoint_root / filename).exists()
    ]


def collect_missing_data_paths(spec: dict[str, Any], categories: list[str]) -> list[str]:
    data_root = spec.get("data_root")
    if data_root is None:
        return []
    missing: list[str] = []
    for rel_path in spec.get("required_data_paths", []):
        if rel_path == "meta_not_required":
            continue
        if not (data_root / rel_path).exists():
            missing.append(str(data_root / rel_path))
    for category in categories:
        category_root = data_root / category
        if not category_root.exists():
            missing.append(str(category_root))
    return missing


def collect_missing_mask_paths(spec: dict[str, Any], categories: list[str]) -> list[str]:
    mask_root = spec.get("mask_root")
    if mask_root is None:
        return []
    missing: list[str] = []
    for category in categories:
        for image_path in collect_univad_category_image_paths(spec, category):
            candidate = expected_univad_mask_path(
                Path(image_path),
                data_root=spec["data_root"],
                mask_root=mask_root,
                category=category,
            )
            if not candidate.exists():
                missing.append(str(candidate))
    return missing


def expected_univad_mask_path(image_path: Path, *, data_root: Path, mask_root: Path, category: str) -> Path:
    category_root = data_root / category
    try:
        rel_path = image_path.relative_to(category_root)
    except ValueError:
        parts = list(image_path.parts)
        if category not in parts:
            raise
        category_index = len(parts) - 1 - parts[::-1].index(category)
        rel_path = Path(*parts[category_index + 1 :])
    return mask_root / category / rel_path.with_suffix("") / "grounding_mask.png"


def build_mask_backup_status(
    *,
    attempted: bool,
    status: str,
    destination_root: Path | None,
    copied_categories: list[str] | None = None,
    copied_mask_files: int = 0,
    note: str = "-",
) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "status": status,
        "destination_root": str(destination_root) if destination_root is not None else "-",
        "copied_categories": copied_categories or [],
        "copied_mask_files": copied_mask_files,
        "note": note,
    }


def build_mask_restore_status(
    *,
    attempted: bool,
    status: str,
    source_root: Path | None,
    copied_categories: list[str] | None = None,
    copied_mask_files: int = 0,
    note: str = "-",
) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "status": status,
        "source_root": str(source_root) if source_root is not None else "-",
        "copied_categories": copied_categories or [],
        "copied_mask_files": copied_mask_files,
        "note": note,
    }


def resolve_univad_mask_drive_root(settings: dict[str, Any]) -> Path:
    return Path(
        os.environ.get(
            "UNIVAD_MASK_DRIVE_BACKUP_ROOT",
            str(settings.get("univad_mask_drive_backup_root", "")),
        )
    )


def maybe_mount_google_drive_for_backup(destination_root: Path, settings: dict[str, Any]) -> None:
    if not str(destination_root).startswith("/content/drive"):
        return
    if Path("/content/drive/MyDrive").exists():
        return
    if not settings.get("auto_mount_google_drive_for_univad_masks", False):
        return
    try:
        from google.colab import drive  # type: ignore[import-not-found]  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return
    print("mount Google Drive for UniVAD mask backup")
    drive.mount("/content/drive")


def maybe_restore_univad_grounding_masks_from_drive(
    spec: dict[str, Any],
    *,
    categories: list[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not settings.get("restore_univad_grounding_masks_from_drive", False):
        return build_mask_restore_status(
            attempted=False,
            status="disabled",
            source_root=None,
            note="UniVAD mask Drive restore disabled",
        )

    source_root = resolve_univad_mask_drive_root(settings)
    if not source_root:
        return build_mask_restore_status(
            attempted=True,
            status="source_unset",
            source_root=None,
            note="UniVAD mask Drive restore root is empty",
        )
    maybe_mount_google_drive_for_backup(source_root, settings)
    if str(source_root).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        return build_mask_restore_status(
            attempted=True,
            status="drive_unavailable",
            source_root=source_root,
            note="Mount Google Drive in Colab before rerunning setup to restore UniVAD masks",
        )
    if not source_root.exists():
        return build_mask_restore_status(
            attempted=True,
            status="source_missing",
            source_root=source_root,
            note=f"UniVAD mask Drive cache does not exist: {source_root}",
        )

    mask_root = spec["mask_root"]
    mask_root.mkdir(parents=True, exist_ok=True)
    copied_categories: list[str] = []
    copied_mask_files = 0
    for category in categories:
        source_category = source_root / category
        if not source_category.exists():
            continue
        destination_category = mask_root / category
        shutil.copytree(source_category, destination_category, dirs_exist_ok=True)
        copied_categories.append(category)
        copied_mask_files += sum(1 for _ in destination_category.rglob("grounding_mask.png"))

    status = "ok" if copied_categories else "no_category_masks"
    note = (
        f"UniVAD grounding masks restored from {source_root}"
        if copied_categories
        else f"No UniVAD category mask folders found under {source_root}"
    )
    return build_mask_restore_status(
        attempted=True,
        status=status,
        source_root=source_root,
        copied_categories=copied_categories,
        copied_mask_files=copied_mask_files,
        note=note,
    )


def maybe_backup_univad_grounding_masks(
    spec: dict[str, Any],
    *,
    categories: list[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not settings.get("backup_univad_grounding_masks_to_drive", False):
        return build_mask_backup_status(
            attempted=False,
            status="disabled",
            destination_root=None,
            note="UniVAD mask Drive backup disabled",
        )

    destination_root = resolve_univad_mask_drive_root(settings)
    if not destination_root:
        return build_mask_backup_status(
            attempted=True,
            status="destination_unset",
            destination_root=None,
            note="UniVAD mask Drive backup root is empty",
        )
    maybe_mount_google_drive_for_backup(destination_root, settings)
    if str(destination_root).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        return build_mask_backup_status(
            attempted=True,
            status="drive_unavailable",
            destination_root=destination_root,
            note="Mount Google Drive in Colab before rerunning setup to persist UniVAD masks",
        )

    mask_root = spec["mask_root"]
    if not mask_root.exists():
        return build_mask_backup_status(
            attempted=True,
            status="source_missing",
            destination_root=destination_root,
            note=f"UniVAD mask root does not exist: {mask_root}",
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    copied_categories: list[str] = []
    copied_mask_files = 0
    for category in categories:
        source_category = mask_root / category
        if not source_category.exists():
            continue
        destination_category = destination_root / category
        shutil.copytree(source_category, destination_category, dirs_exist_ok=True)
        copied_categories.append(category)
        copied_mask_files += sum(1 for _ in destination_category.rglob("grounding_mask.png"))

    status = "ok" if copied_categories else "no_category_masks"
    note = (
        f"UniVAD grounding masks backed up to {destination_root}"
        if copied_categories
        else f"No UniVAD category mask folders found under {mask_root}"
    )
    return build_mask_backup_status(
        attempted=True,
        status=status,
        destination_root=destination_root,
        copied_categories=copied_categories,
        copied_mask_files=copied_mask_files,
        note=note,
    )


def evaluate_baseline_readiness(
    active_baselines: list[str],
    *,
    specs: dict[str, dict[str, Any]],
    categories: list[str],
) -> pd.DataFrame:
    import torch  # noqa: WPS433

    rows: list[dict[str, Any]] = []
    cuda_available = torch.cuda.is_available()
    for baseline in active_baselines:
        spec = specs[baseline]
        external_dir = spec["external_dir"]
        repo_ready = external_dir.exists()
        checkpoint_root = spec.get("checkpoint_root")
        missing_checkpoint_files = collect_missing_checkpoint_files(spec)
        checkpoint_ready = len(missing_checkpoint_files) == 0
        missing_local_paths = [
            str(path) for path in spec.get("required_local_paths", []) if not Path(path).exists()
        ]
        local_paths_ready = len(missing_local_paths) == 0
        missing_data_paths = collect_missing_data_paths(spec, categories)
        data_ready = len(missing_data_paths) == 0
        missing_mask_paths = collect_missing_mask_paths(spec, categories)
        masks_ready = len(missing_mask_paths) == 0
        cuda_ready = cuda_available if spec.get("requires_cuda") else True
        for import_path in spec.get("import_search_paths", []):
            ensure_importable_path(import_path)
        missing_modules = [
            module_name
            for module_name in spec.get("required_python_modules", [])
            if importlib.util.find_spec(module_name) is None
        ]
        python_modules_ready = len(missing_modules) == 0
        ready = (
            repo_ready
            and checkpoint_ready
            and local_paths_ready
            and data_ready
            and masks_ready
            and cuda_ready
            and python_modules_ready
        )
        blockers: list[str] = []
        if not repo_ready:
            blockers.append("external_repo_missing")
        if spec.get("requires_cuda") and not cuda_ready:
            blockers.append("cuda_unavailable")
        if not local_paths_ready:
            blockers.append("local_dependency_missing:" + ",".join(Path(path).name for path in missing_local_paths))
        if not data_ready:
            blockers.append("dataset_missing:" + ",".join(Path(path).name for path in missing_data_paths[:4]))
        if not masks_ready:
            missing_preview = [str(Path(path)) for path in missing_mask_paths[:4]]
            missing_suffix = f"+{len(missing_mask_paths) - 4} more" if len(missing_mask_paths) > 4 else ""
            blockers.append(
                "grounding_masks_missing:"
                + ",".join(missing_preview)
                + (f",{missing_suffix}" if missing_suffix else "")
            )
        if spec.get("requires_checkpoints") and not checkpoint_ready:
            blockers.append("checkpoint_missing:" + ",".join(missing_checkpoint_files))
        if missing_modules:
            blockers.append("python_module_missing:" + ",".join(missing_modules))
        rows.append(
            {
                "baseline": baseline,
                "requested": True,
                "device": spec["device"],
                "data_root": str(spec.get("data_root", "")),
                "external_repo_exists": repo_ready,
                "cuda_required": spec.get("requires_cuda", False),
                "cuda_available": cuda_available,
                "checkpoint_required": spec.get("requires_checkpoints", False),
                "checkpoint_root": str(checkpoint_root) if checkpoint_root else "",
                "checkpoint_ready": checkpoint_ready,
                "missing_checkpoint_files": ", ".join(missing_checkpoint_files) if missing_checkpoint_files else "-",
                "missing_local_paths": ", ".join(missing_local_paths) if missing_local_paths else "-",
                "data_ready": data_ready,
                "missing_data_paths": ", ".join(missing_data_paths) if missing_data_paths else "-",
                "mask_root": str(spec.get("mask_root", "")),
                "masks_ready": masks_ready,
                "missing_mask_paths": (
                    ", ".join(missing_mask_paths[:10])
                    + (f" ... +{len(missing_mask_paths) - 10} more" if len(missing_mask_paths) > 10 else "")
                    if missing_mask_paths
                    else "-"
                ),
                "required_python_modules": ", ".join(spec.get("required_python_modules", [])) or "-",
                "python_modules_ready": python_modules_ready,
                "ready": ready,
                "blockers": ", ".join(blockers) if blockers else "ready",
                "notes": spec["notes"],
            }
        )
    return pd.DataFrame(rows)


def build_local_dependency_status_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "baseline": "UniVAD",
            "required_path": str(path),
            "exists": Path(path).exists(),
            "path": str(path),
        }
        for path in spec.get("required_local_paths", [])
    ]


def build_checkpoint_status_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoint_root = spec.get("checkpoint_root")
    if checkpoint_root is None:
        return []
    return [
        {
            "baseline": "UniVAD",
            "checkpoint_file": filename,
            "exists": (checkpoint_root / filename).exists(),
            "path": str(checkpoint_root / filename),
        }
        for filename in spec.get("required_checkpoint_files", [])
    ]


def build_dataset_status_rows(spec: dict[str, Any], categories: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data_root = spec.get("data_root")
    for rel_path in spec.get("required_data_paths", []):
        if rel_path == "meta_not_required":
            continue
        candidate = data_root / rel_path
        rows.append(
            {
                "baseline": "UniVAD",
                "required_data_path": str(candidate),
                "exists": candidate.exists(),
                "path": str(candidate),
            }
        )
    for category in categories:
        category_root = data_root / category
        rows.append(
            {
                "baseline": "UniVAD",
                "required_data_path": str(category_root),
                "exists": category_root.exists(),
                "path": str(category_root),
            }
        )
    return rows


def build_mask_status_rows(spec: dict[str, Any], categories: list[str]) -> list[dict[str, Any]]:
    mask_root = spec.get("mask_root")
    if mask_root is None:
        return []
    rows: list[dict[str, Any]] = []
    for category in categories:
        image_paths = collect_univad_category_image_paths(spec, category)
        expected_paths = [
            expected_univad_mask_path(
                Path(image_path),
                data_root=spec["data_root"],
                mask_root=mask_root,
                category=category,
            )
            for image_path in image_paths
        ]
        missing_paths = [path for path in expected_paths if not path.exists()]
        rows.append(
            {
                "baseline": "UniVAD",
                "category": category,
                "image_count": len(image_paths),
                "mask_count": len(expected_paths) - len(missing_paths),
                "missing_mask_count": len(missing_paths),
                "sample_missing_mask_path": str(missing_paths[0]) if missing_paths else "-",
                "mask_root": str(mask_root / category),
                "exists": len(missing_paths) == 0,
            }
        )
    return rows


def ensure_git_repo(repo_dir: Path, repo_url: str, *, recurse_submodules: bool = False) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_flag = "--recurse-submodules " if recurse_submodules else ""
    subprocess.run(
        ["bash", "-lc", f'test -d "{repo_dir}/.git" || git clone {clone_flag}"{repo_url}" "{repo_dir}"'],
        check=True,
    )
    if recurse_submodules:
        subprocess.run(["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"], check=True)


def ensure_editable_package(package_dir: Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(package_dir)], check=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    ensure_directory(destination.parent)
    subprocess.run(["curl", "-L", "--fail", "-o", str(destination), url], check=True)


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def run_pip_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"run pip: {' '.join(args)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
    return result


def summarize_command_failure(result: subprocess.CompletedProcess[str]) -> str:
    output_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    output_tail = " | ".join(output_lines[-8:]) if output_lines else "no pip output captured"
    command = " ".join(str(part) for part in result.args)
    return f"command failed with exit code {result.returncode}: {command}; {output_tail}"


@contextlib.contextmanager
def working_directory(path: Path):
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def patch_univad_component_segmentation(univad_dir: Path) -> bool:
    component_path = univad_dir / "models" / "component_segmentaion.py"
    target = "        if len(boxes_filt) == 0:\n            masks = np.ones((W, H))\n"
    replacement = (
        "        if len(boxes_filt) == 0:\n"
        "            background = np.zeros((H, W), dtype=np.uint8)\n"
        "            masks = np.ones((H, W), dtype=np.uint8) * 255\n"
        "            color_mask = masks.copy()\n"
    )
    text = component_path.read_text(encoding="utf-8")
    if replacement in text:
        return False
    if target not in text:
        print(f"warning: could not find expected zero-box branch in {component_path}")
        return False
    component_path.write_text(text.replace(target, replacement), encoding="utf-8")
    print(f"patched UniVAD zero-box grounding branch: {component_path}")
    return True


def patch_univad_optional_pydensecrf(univad_dir: Path) -> bool:
    crf_path = univad_dir / "utils" / "crf.py"
    if not crf_path.exists():
        print(f"warning: UniVAD CRF file not found: {crf_path}")
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
    if import_target not in text:
        print(f"warning: could not find expected pydensecrf imports in {crf_path}")
        return False
    text = text.replace(import_target, import_replacement)

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
    if function_target not in text:
        print(f"warning: could not find expected dense_crf body in {crf_path}")
        return False
    crf_path.write_text(text.replace(function_target, function_replacement), encoding="utf-8")
    print(f"patched UniVAD optional pydensecrf fallback: {crf_path}")
    return True


def inspect_univad_torch_stack() -> dict[str, Any]:
    status = {
        "ready": False,
        "restart_required": False,
        "torch_version": "-",
        "torchvision_version": "-",
        "torchaudio_version": "-",
        "cuda_available": False,
        "error": "-",
        "note": "-",
    }
    try:
        torch_mod = importlib.import_module("torch")
        status["torch_version"] = getattr(torch_mod, "__version__", "-")
        status["cuda_available"] = bool(torch_mod.cuda.is_available())
        torchvision_mod = importlib.import_module("torchvision")
        torchaudio_mod = importlib.import_module("torchaudio")
        from torchvision.transforms.functional import resize as _tv_resize  # noqa: F401,WPS433

        status["torchvision_version"] = getattr(torchvision_mod, "__version__", "-")
        status["torchaudio_version"] = getattr(torchaudio_mod, "__version__", "-")
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["note"] = "torch stack import failed"
        return status
    version_match = (
        status["torch_version"].startswith(UNIVAD_TORCH_VERSION)
        and status["torchvision_version"].startswith(UNIVAD_TORCHVISION_VERSION)
        and status["torchaudio_version"].startswith(UNIVAD_TORCHAUDIO_VERSION)
    )
    status["ready"] = bool(version_match)
    status["note"] = "torch stack compatible" if version_match else "torch stack version mismatch"
    return status


def maybe_fix_univad_torch_stack(settings: dict[str, Any]) -> dict[str, Any]:
    status = inspect_univad_torch_stack()
    if status["ready"] or not settings["auto_fix_univad_torch_stack"]:
        return status
    print("install compatible torch stack for UniVAD")
    result = run_pip_command(
        [
            "install",
            "--upgrade",
            "--no-cache-dir",
            f"torch=={UNIVAD_TORCH_VERSION}",
            f"torchvision=={UNIVAD_TORCHVISION_VERSION}",
            f"torchaudio=={UNIVAD_TORCHAUDIO_VERSION}",
        ]
    )
    if result.returncode != 0:
        status["error"] = summarize_command_failure(result)
        status["note"] = "UniVAD torch stack install failed; fix the pip error above and rerun setup"
        return status
    status["restart_required"] = True
    status["note"] = "installed compatible torch stack; restart runtime and rerun notebook from the top"
    return status


def install_univad_requirements_without_torch(requirements_path: Path) -> Path:
    blocked_prefixes = ("torch", "torchvision", "torchaudio", "transformers", "numpy", "opencv")
    filtered_lines: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            filtered_lines.append(raw_line)
            continue
        normalized = stripped.lower().replace("_", "-").replace(" ", "")
        if normalized.startswith(blocked_prefixes):
            continue
        filtered_lines.append(raw_line)
    temp_requirements = Path("/tmp/univad_requirements_notorch.txt")
    temp_requirements.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(temp_requirements)], check=True)
    return temp_requirements


def inspect_univad_runtime_dependency_stack() -> dict[str, Any]:
    status = {
        "ready": False,
        "restart_required": False,
        "numpy_version": "-",
        "opencv_version": "-",
        "torchao_present": False,
        "missing_modules": [],
        "error": "-",
        "note": "-",
    }
    try:
        numpy_mod = importlib.import_module("numpy")
        status["numpy_version"] = getattr(numpy_mod, "__version__", "-")
        cv2_mod = importlib.import_module("cv2")
        status["opencv_version"] = getattr(cv2_mod, "__version__", "-")
        status["torchao_present"] = importlib.util.find_spec("torchao") is not None
        for module_name in UNIVAD_RUNTIME_DEPENDENCIES:
            try:
                importlib.import_module(module_name)
            except Exception:  # noqa: BLE001
                status["missing_modules"].append(module_name)
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["note"] = "UniVAD runtime dependency import failed"
        return status
    numpy_ready = status["numpy_version"].startswith("1.")
    opencv_ready = status["opencv_version"].startswith("4.11.") or status["opencv_version"].startswith("4.10.")
    if numpy_ready and opencv_ready and not status["torchao_present"] and not status["missing_modules"]:
        status["ready"] = True
        status["note"] = "UniVAD runtime dependencies compatible"
    else:
        status["note"] = "UniVAD runtime dependency mismatch or missing module"
    return status


def maybe_fix_univad_runtime_dependency_stack(settings: dict[str, Any]) -> dict[str, Any]:
    status = inspect_univad_runtime_dependency_stack()
    if status["ready"] or not settings["auto_fix_univad_runtime_deps"]:
        return status
    print("install compatible UniVAD runtime dependency stack")
    run_pip_command(["uninstall", "-y", "torchao", "torchao-nightly"])
    run_pip_command(
        [
            "uninstall",
            "-y",
            "opencv-python",
            "opencv-python-headless",
            "opencv-contrib-python",
            "opencv-contrib-python-headless",
        ]
    )
    install_steps = [
        ["install", "--upgrade", "--no-cache-dir", f"numpy=={UNIVAD_NUMPY_VERSION}"],
        ["install", "--upgrade", "--no-cache-dir", f"opencv-python-headless=={UNIVAD_OPENCV_VERSION}"],
        [
            "install",
            "--upgrade",
            "--no-cache-dir",
            *UNIVAD_RUNTIME_DEPENDENCY_SPECS,
            f"numpy=={UNIVAD_NUMPY_VERSION}",
        ],
    ]
    for install_args in install_steps:
        result = run_pip_command(install_args)
        if result.returncode != 0:
            status["error"] = summarize_command_failure(result)
            status["note"] = "UniVAD runtime dependency install failed; fix the pip error above and rerun setup"
            return status
    status["restart_required"] = True
    status["note"] = (
        "installed compatible UniVAD runtime deps with headless OpenCV; "
        "restart runtime and rerun notebook from the top"
    )
    return status


def inspect_univad_transformers_stack() -> dict[str, Any]:
    status = {
        "ready": False,
        "restart_required": False,
        "transformers_version": "-",
        "bert_has_get_head_mask": False,
        "error": "-",
        "note": "-",
    }
    try:
        transformers_mod = importlib.import_module("transformers")
        from transformers import BertModel  # noqa: WPS433

        status["transformers_version"] = getattr(transformers_mod, "__version__", "-")
        status["bert_has_get_head_mask"] = hasattr(BertModel, "get_head_mask")
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["note"] = "transformers import failed"
        return status
    version_match = status["transformers_version"].startswith(UNIVAD_TRANSFORMERS_VERSION)
    if version_match and status["bert_has_get_head_mask"]:
        status["ready"] = True
        status["note"] = "transformers stack compatible"
    else:
        status["note"] = "transformers stack version mismatch or missing BertModel.get_head_mask"
    return status


def maybe_fix_univad_transformers_stack(settings: dict[str, Any]) -> dict[str, Any]:
    status = inspect_univad_transformers_stack()
    if status["ready"] or not settings["auto_fix_univad_transformers_stack"]:
        return status
    print("install compatible transformers stack for UniVAD/GroundingDINO")
    result = run_pip_command(
        [
            "install",
            "--upgrade",
            "--no-cache-dir",
            f"transformers=={UNIVAD_TRANSFORMERS_VERSION}",
        ]
    )
    if result.returncode != 0:
        status["error"] = summarize_command_failure(result)
        status["note"] = "UniVAD transformers stack install failed; fix the pip error above and rerun setup"
        return status
    status["restart_required"] = True
    status["note"] = "installed compatible transformers stack; restart runtime and rerun notebook from the top"
    return status


def maybe_download_univad_checkpoints(spec: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    checkpoint_root = spec["checkpoint_root"]
    ensure_directory(checkpoint_root)
    downloaded: list[str] = []
    if not settings["download_missing_univad_checkpoints"]:
        return downloaded
    for filename in collect_missing_checkpoint_files(spec):
        url = spec["checkpoint_download_urls"].get(filename)
        if not url:
            continue
        destination = checkpoint_root / filename
        print(f"download checkpoint: {filename} <- {url}")
        download_file(url, destination)
        downloaded.append(filename)
    return downloaded


def maybe_prepare_univad_caption_dataset(
    spec: dict[str, Any],
    *,
    raw_loco_root: Path,
    settings: dict[str, Any],
) -> bool:
    data_root = spec["data_root"]
    meta_path = data_root / "meta.json"
    if meta_path.exists() or not settings["auto_prepare_univad_caption_dataset"]:
        return False
    if not raw_loco_root.exists():
        return False
    prepare_script = spec["caption_prepare_script"]
    print(f"prepare UniVAD caption dataset: {data_root} <- {raw_loco_root}")
    subprocess.run(
        [sys.executable, str(prepare_script), "--src-root", str(raw_loco_root), "--dst-root", str(data_root)],
        check=True,
    )
    return True


def collect_univad_category_image_paths(spec: dict[str, Any], category: str) -> list[str]:
    category_root = spec["data_root"] / category
    image_paths: list[str] = []
    for split in ("train", "test"):
        split_root = category_root / split
        if split_root.exists():
            image_paths.extend(str(path) for path in sorted(split_root.rglob("*.png")))
    return image_paths


def maybe_prepare_univad_grounding_masks(
    spec: dict[str, Any],
    *,
    categories: list[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    import torch  # noqa: WPS433

    status = {
        "attempted": False,
        "generated": False,
        "generated_categories": [],
        "patched_component_segmentation": False,
        "error": "-",
        "traceback": "-",
        "reason": "disabled",
    }
    if not settings["auto_prepare_univad_grounding_masks"]:
        return status
    if spec.get("requires_cuda") and not torch.cuda.is_available():
        status["reason"] = "cuda_unavailable"
        return status
    if collect_missing_data_paths(spec, categories):
        status["reason"] = "dataset_missing"
        return status
    if not collect_missing_mask_paths(spec, categories):
        status["reason"] = "already_present"
        return status

    univad_dir = spec["external_dir"]
    local_data_dir = univad_dir / "data" / "mvtec_loco_caption"
    ensure_symlink(spec["data_root"], local_data_dir)
    status["attempted"] = True
    status["reason"] = "running"

    try:
        status["patched_component_segmentation"] = patch_univad_component_segmentation(univad_dir)
        ensure_importable_path(univad_dir)
        ensure_importable_path(spec["groundingdino_dir"])
        with working_directory(univad_dir):
            from models.component_segmentaion import grounding_segmentation  # noqa: WPS433
            import yaml  # noqa: WPS433

            for category in categories:
                image_paths = collect_univad_category_image_paths(spec, category)
                if not image_paths:
                    continue
                config_path = univad_dir / "configs" / "class_histogram" / f"{category}.yaml"
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                output_dir = spec["mask_root"] / category
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"generate UniVAD grounding masks: category={category} images={len(image_paths)} output={output_dir}")
                grounding_segmentation(image_paths, str(output_dir), config["grounding_config"])
                status["generated_categories"].append(category)
        status["generated"] = bool(status["generated_categories"])
        status["reason"] = "ok" if status["generated"] else "no_images"
        return status
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc(limit=12)
        status["reason"] = "failed"
        print("UniVAD grounding mask generation failed")
        print(status["traceback"])
        return status


def setup_patchcore(spec: dict[str, Any]) -> dict[str, Any]:
    patchcore_dir = spec["external_dir"]
    ensure_git_repo(patchcore_dir, spec["external_repo_url"])
    subprocess.run([sys.executable, "-m", "pip", "install", "faiss-cpu", "timm"], check=True)
    return {
        "baseline": "PatchCore",
        "external_dir": str(patchcore_dir),
        "data_root": str(spec["data_root"]),
        "setup_status": "ready",
        "note": "Repo synced and lightweight runtime deps installed.",
    }


def build_univad_setup_blocked_row(
    spec: dict[str, Any],
    *,
    univad_dir: Path,
    groundingdino_dir: Path,
    checkpoint_root: Path,
    missing_local_paths: list[str],
    caption_dataset_prepared: bool,
    downloaded_checkpoint_files: list[str],
    mask_restore_status: dict[str, Any],
    mask_backup_status: dict[str, Any],
    setup_status: str,
    mask_generation_reason: str,
    mask_generation_error: str,
    note: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "baseline": "UniVAD",
        "external_dir": str(univad_dir),
        "data_root": str(spec["data_root"]),
        "mask_root": str(spec["mask_root"]),
        "groundingdino_dir": str(groundingdino_dir),
        "checkpoint_root": str(checkpoint_root),
        "groundingdino_importable": False,
        "missing_local_paths": ", ".join(missing_local_paths) if missing_local_paths else "-",
        "caption_dataset_prepared": caption_dataset_prepared,
        "masks_generated": False,
        "mask_generation_attempted": False,
        "mask_generation_reason": mask_generation_reason,
        "mask_generation_error": mask_generation_error,
        "mask_generation_categories": "-",
        "patched_component_segmentation": False,
        "download_missing_checkpoints": settings["download_missing_univad_checkpoints"],
        "downloaded_checkpoint_files": ", ".join(downloaded_checkpoint_files) if downloaded_checkpoint_files else "-",
        "missing_checkpoint_files": ", ".join(collect_missing_checkpoint_files(spec)) or "-",
        "mask_restore_status": mask_restore_status["status"],
        "mask_restore_path": mask_restore_status["source_root"],
        "mask_restore_categories": (
            ", ".join(mask_restore_status["copied_categories"])
            if mask_restore_status["copied_categories"]
            else "-"
        ),
        "mask_restore_file_count": mask_restore_status["copied_mask_files"],
        "mask_backup_status": mask_backup_status["status"],
        "mask_backup_path": mask_backup_status["destination_root"],
        "mask_backup_categories": (
            ", ".join(mask_backup_status["copied_categories"])
            if mask_backup_status["copied_categories"]
            else "-"
        ),
        "mask_backup_file_count": mask_backup_status["copied_mask_files"],
        "missing_data_paths": "-",
        "missing_mask_paths": "-",
        "setup_status": setup_status,
        "note": note,
    }


def _blocked_from_status(
    status: dict[str, Any],
    spec: dict[str, Any],
    *,
    univad_dir: Path,
    groundingdino_dir: Path,
    checkpoint_root: Path,
    missing_local_paths: list[str],
    caption_dataset_prepared: bool,
    downloaded_checkpoint_files: list[str],
    mask_restore_status: dict[str, Any],
    mask_backup_status: dict[str, Any],
    unavailable_reason: str,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    if status.get("restart_required"):
        return build_univad_setup_blocked_row(
            spec,
            univad_dir=univad_dir,
            groundingdino_dir=groundingdino_dir,
            checkpoint_root=checkpoint_root,
            missing_local_paths=missing_local_paths,
            caption_dataset_prepared=caption_dataset_prepared,
            downloaded_checkpoint_files=downloaded_checkpoint_files,
            mask_restore_status=mask_restore_status,
            mask_backup_status=mask_backup_status,
            setup_status="restart_required",
            mask_generation_reason="runtime_restart_required",
            mask_generation_error="-",
            note=status["note"],
            settings=settings,
        )
    if not status.get("ready"):
        return build_univad_setup_blocked_row(
            spec,
            univad_dir=univad_dir,
            groundingdino_dir=groundingdino_dir,
            checkpoint_root=checkpoint_root,
            missing_local_paths=missing_local_paths,
            caption_dataset_prepared=caption_dataset_prepared,
            downloaded_checkpoint_files=downloaded_checkpoint_files,
            mask_restore_status=mask_restore_status,
            mask_backup_status=mask_backup_status,
            setup_status="blocked",
            mask_generation_reason=unavailable_reason,
            mask_generation_error=status["error"],
            note=status["note"],
            settings=settings,
        )
    return None


def setup_univad(
    spec: dict[str, Any],
    *,
    categories: list[str],
    raw_loco_root: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    univad_dir = spec["external_dir"]
    ensure_git_repo(univad_dir, spec["external_repo_url"], recurse_submodules=True)
    patched_optional_pydensecrf = patch_univad_optional_pydensecrf(univad_dir)
    requirements_path = univad_dir / "requirements.txt"
    groundingdino_dir = spec["groundingdino_dir"]
    checkpoint_root = spec["checkpoint_root"]
    missing_local_paths = [str(path) for path in spec.get("required_local_paths", []) if not Path(path).exists()]
    ensure_directory(checkpoint_root)
    caption_dataset_prepared = maybe_prepare_univad_caption_dataset(
        spec, raw_loco_root=raw_loco_root, settings=settings
    )
    downloaded_checkpoint_files = maybe_download_univad_checkpoints(spec, settings)
    mask_restore_status = maybe_restore_univad_grounding_masks_from_drive(
        spec, categories=categories, settings=settings
    )
    mask_backup_status = maybe_backup_univad_grounding_masks(spec, categories=categories, settings=settings)

    dependency_checks = [
        (maybe_fix_univad_runtime_dependency_stack, "runtime_dependency_unavailable"),
        (maybe_fix_univad_torch_stack, "torch_stack_unavailable"),
        (maybe_fix_univad_transformers_stack, "transformers_stack_unavailable"),
    ]
    for check_dependency_stack, unavailable_reason in dependency_checks:
        status = check_dependency_stack(settings)
        blocked_row = _blocked_from_status(
            status,
            spec,
            univad_dir=univad_dir,
            groundingdino_dir=groundingdino_dir,
            checkpoint_root=checkpoint_root,
            missing_local_paths=missing_local_paths,
            caption_dataset_prepared=caption_dataset_prepared,
            downloaded_checkpoint_files=downloaded_checkpoint_files,
            mask_restore_status=mask_restore_status,
            mask_backup_status=mask_backup_status,
            unavailable_reason=unavailable_reason,
            settings=settings,
        )
        if blocked_row is not None:
            return blocked_row

    if requirements_path.exists():
        filtered_requirements_path = install_univad_requirements_without_torch(requirements_path)
        requirement_note = f"requirements installed without torch/transformers/numpy/opencv stack: {filtered_requirements_path}"
    else:
        requirement_note = "requirements.txt not found; using current runtime packages"

    ensure_editable_package(groundingdino_dir)
    ensure_importable_path(groundingdino_dir)
    mask_status = maybe_prepare_univad_grounding_masks(spec, categories=categories, settings=settings)
    mask_backup_status = maybe_backup_univad_grounding_masks(spec, categories=categories, settings=settings)
    missing_checkpoint_files = collect_missing_checkpoint_files(spec)
    missing_data_paths = collect_missing_data_paths(spec, categories)
    missing_mask_paths = collect_missing_mask_paths(spec, categories)
    missing_mask_paths_display = (
        ", ".join(missing_mask_paths[:10])
        + (f" ... +{len(missing_mask_paths) - 10} more" if len(missing_mask_paths) > 10 else "")
        if missing_mask_paths
        else "-"
    )
    return {
        "baseline": "UniVAD",
        "external_dir": str(univad_dir),
        "data_root": str(spec["data_root"]),
        "mask_root": str(spec["mask_root"]),
        "groundingdino_dir": str(groundingdino_dir),
        "checkpoint_root": str(checkpoint_root),
        "groundingdino_importable": module_available("groundingdino", extra_paths=[groundingdino_dir]),
        "missing_local_paths": ", ".join(missing_local_paths) if missing_local_paths else "-",
        "caption_dataset_prepared": caption_dataset_prepared,
        "masks_generated": mask_status["generated"],
        "mask_generation_attempted": mask_status["attempted"],
        "mask_generation_reason": mask_status["reason"],
        "mask_generation_error": mask_status["error"],
        "mask_generation_categories": (
            ", ".join(mask_status["generated_categories"]) if mask_status["generated_categories"] else "-"
        ),
        "patched_optional_pydensecrf": patched_optional_pydensecrf,
        "patched_component_segmentation": mask_status["patched_component_segmentation"],
        "download_missing_checkpoints": settings["download_missing_univad_checkpoints"],
        "downloaded_checkpoint_files": ", ".join(downloaded_checkpoint_files) if downloaded_checkpoint_files else "-",
        "missing_checkpoint_files": ", ".join(missing_checkpoint_files) if missing_checkpoint_files else "-",
        "mask_restore_status": mask_restore_status["status"],
        "mask_restore_path": mask_restore_status["source_root"],
        "mask_restore_categories": (
            ", ".join(mask_restore_status["copied_categories"])
            if mask_restore_status["copied_categories"]
            else "-"
        ),
        "mask_restore_file_count": mask_restore_status["copied_mask_files"],
        "mask_backup_status": mask_backup_status["status"],
        "mask_backup_path": mask_backup_status["destination_root"],
        "mask_backup_categories": (
            ", ".join(mask_backup_status["copied_categories"])
            if mask_backup_status["copied_categories"]
            else "-"
        ),
        "mask_backup_file_count": mask_backup_status["copied_mask_files"],
        "missing_data_paths": ", ".join(missing_data_paths) if missing_data_paths else "-",
        "missing_mask_paths": missing_mask_paths_display,
        "setup_status": "ready" if mask_status["error"] == "-" else "partial",
        "note": requirement_note,
    }


def setup_baseline(
    baseline: str,
    *,
    specs: dict[str, dict[str, Any]],
    categories: list[str],
    raw_loco_root: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    spec = specs[baseline]
    if baseline == "PatchCore":
        return setup_patchcore(spec)
    if baseline == "UniVAD":
        return setup_univad(spec, categories=categories, raw_loco_root=raw_loco_root, settings=settings)
    raise ValueError(f"Unsupported baseline setup={baseline}")


def safe_setup_baseline(
    baseline: str,
    *,
    specs: dict[str, dict[str, Any]],
    categories: list[str],
    raw_loco_root: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    try:
        return setup_baseline(
            baseline,
            specs=specs,
            categories=categories,
            raw_loco_root=raw_loco_root,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"baseline setup failed: {baseline}")
        print(traceback.format_exc(limit=12))
        return {
            "baseline": baseline,
            "setup_status": "failed",
            "setup_error": f"{type(exc).__name__}: {exc}",
            "note": "See notebook output above for traceback.",
        }


def run_baseline_setup(
    *,
    requested_baselines: list[str],
    specs: dict[str, dict[str, Any]],
    categories: list[str],
    raw_loco_root: Path,
    settings: dict[str, Any],
    requested_run_configs: list[dict[str, Any]],
    run_only_ready_baselines: bool,
) -> dict[str, Any]:
    setup_rows = [
        safe_setup_baseline(
            baseline,
            specs=specs,
            categories=categories,
            raw_loco_root=raw_loco_root,
            settings=settings,
        )
        for baseline in requested_baselines
    ]
    baseline_status_df = evaluate_baseline_readiness(requested_baselines, specs=specs, categories=categories)
    ready_baselines = baseline_status_df.loc[baseline_status_df["ready"], "baseline"].tolist()
    if run_only_ready_baselines:
        run_configs = [config for config in requested_run_configs if config["baseline"] in ready_baselines]
        skipped_run_configs = [config for config in requested_run_configs if config["baseline"] not in ready_baselines]
    else:
        run_configs = list(requested_run_configs)
        skipped_run_configs = []
    baseline_status_df["will_run"] = baseline_status_df["baseline"].isin(
        {config["baseline"] for config in run_configs}
    )
    return {
        "setup_df": pd.DataFrame(setup_rows),
        "baseline_status_df": baseline_status_df,
        "run_configs": run_configs,
        "skipped_run_configs": skipped_run_configs,
        "univad_dataset_status_df": pd.DataFrame(build_dataset_status_rows(specs["UniVAD"], categories))
        if "UniVAD" in requested_baselines
        else pd.DataFrame(),
        "univad_local_dependency_status_df": pd.DataFrame(build_local_dependency_status_rows(specs["UniVAD"]))
        if "UniVAD" in requested_baselines
        else pd.DataFrame(),
        "univad_checkpoint_status_df": pd.DataFrame(build_checkpoint_status_rows(specs["UniVAD"]))
        if "UniVAD" in requested_baselines
        else pd.DataFrame(),
        "univad_mask_status_df": pd.DataFrame(build_mask_status_rows(specs["UniVAD"], categories))
        if "UniVAD" in requested_baselines
        else pd.DataFrame(),
    }
