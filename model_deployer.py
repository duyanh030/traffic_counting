import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config


MODELS_DIR = config.BASE_DIR / "models"
STAGING_DIR = MODELS_DIR / "staging"
VERSION_JSON_PATH = MODELS_DIR / "version.json"


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def get_active_model_info() -> Dict[str, Any]:
    """Retrieve currently deployed active model info from version.json."""
    ensure_dirs()
    if not VERSION_JSON_PATH.exists():
        return {
            "active_version": "v1.0.0",
            "model_path": config.MODEL_NAME,
            "created_at": "Initial",
            "history": [],
        }

    with open(VERSION_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_next_version(current_version: str) -> str:
    """Calculate next semantic minor version (e.g. v1.0.0 -> v1.1.0)."""
    try:
        clean_v = current_version.lstrip("v")
        parts = list(map(int, clean_v.split(".")))
        parts[1] += 1
        parts[2] = 0
        return f"v{parts[0]}.{parts[1]}.{parts[2]}"
    except Exception:
        return "v1.1.0"


def stage_model(
    best_weights_path: Path,
    metrics: Dict[str, float],
    samples_count: int,
    base_model: str,
    epochs: int,
) -> Path:
    """Stage a newly trained model candidate awaiting human review on the UI."""
    ensure_dirs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cand_id = f"staging_{timestamp}"
    target_dir = STAGING_DIR / cand_id
    target_dir.mkdir(parents=True, exist_ok=True)

    dest_weights = target_dir / "best.pt"
    shutil.copy2(best_weights_path, dest_weights)

    metadata = {
        "candidate_id": cand_id,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "training_samples": samples_count,
        "metrics": metrics,
        "base_model": str(base_model),
        "status": "pending_approval",
    }

    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return target_dir


def list_staged_models() -> List[Dict[str, Any]]:
    """List all staged models awaiting approval."""
    ensure_dirs()
    staged = []
    for cand_folder in sorted(STAGING_DIR.iterdir(), reverse=True):
        if not cand_folder.is_dir():
            continue
        meta_file = cand_folder / "metadata.json"
        weights_file = cand_folder / "best.pt"
        if meta_file.exists() and weights_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_folder_path"] = str(cand_folder)
                staged.append(data)
            except Exception:
                pass
    return staged


def approve_and_deploy_model(candidate_id_or_folder: str) -> Dict[str, Any]:
    """
    Approve a staged candidate model, promote it to production,
    increment semantic version, and point yolov8_traffic_latest.pt to it.
    """
    ensure_dirs()
    if os.path.isdir(candidate_id_or_folder):
        cand_folder = Path(candidate_id_or_folder)
    else:
        cand_folder = STAGING_DIR / candidate_id_or_folder

    meta_file = cand_folder / "metadata.json"
    weights_file = cand_folder / "best.pt"

    if not meta_file.exists() or not weights_file.exists():
        raise FileNotFoundError(f"Staged candidate files not found at: {cand_folder}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. Determine next version
    active_data = get_active_model_info()
    curr_ver = active_data.get("active_version", "v1.0.0")
    next_ver = get_next_version(curr_ver)

    new_model_filename = f"yolov8_traffic_{next_ver}.pt"
    promoted_path = MODELS_DIR / new_model_filename
    latest_pointer = MODELS_DIR / "yolov8_traffic_latest.pt"

    # 2. Copy artifacts
    shutil.copy2(weights_file, promoted_path)
    shutil.copy2(weights_file, latest_pointer)

    # 3. Update version registry
    deploy_time = time.strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        "version": next_ver,
        "model_file": new_model_filename,
        "deployed_at": deploy_time,
        "training_samples": meta.get("training_samples", 0),
        "metrics": meta.get("metrics", {}),
        "status": "active",
        "staged_from": meta.get("candidate_id"),
    }

    active_data["active_version"] = next_ver
    active_data["model_path"] = str(promoted_path.relative_to(config.BASE_DIR))
    active_data["updated_at"] = deploy_time
    active_data.setdefault("history", []).append(history_entry)

    with open(VERSION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(active_data, f, indent=2)

    # 4. Clean up / mark staging candidate as deployed
    meta["status"] = "deployed"
    meta["deployed_version"] = next_ver
    meta["deployed_at"] = deploy_time
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Remove candidate staging directory once deployed
    shutil.rmtree(cand_folder, ignore_errors=True)

    return {
        "version": next_ver,
        "model_path": str(promoted_path),
        "metrics": meta.get("metrics", {}),
    }


def reject_staged_model(candidate_id_or_folder: str):
    """Discard a staged model candidate."""
    ensure_dirs()
    if os.path.isdir(candidate_id_or_folder):
        cand_folder = Path(candidate_id_or_folder)
    else:
        cand_folder = STAGING_DIR / candidate_id_or_folder

    if cand_folder.exists() and cand_folder.is_dir():
        shutil.rmtree(cand_folder, ignore_errors=True)
