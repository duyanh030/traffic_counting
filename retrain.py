import argparse
import json
import os
import shutil
import time
from pathlib import Path
import torch
from ultralytics import YOLO

import config
from dataset_manager import DatasetManager
from model_deployer import stage_model, approve_and_deploy_model
from golden_test_manager import run_regression_test, load_golden_cases


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Automated Retraining Pipeline & Quality Gate (Step 4)"
    )
    parser.add_argument(
        "--zip",
        type=str,
        default=None,
        help="Path to new CVAT exported dataset zip file (e.g. task_traffic_yolo_1.1.zip)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of fine-tuning epochs (default: 30)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Batch size (0 = auto-select based on device memory, default: 0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for fine-tuning (default: 0.001)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Starting model weights (default: models/yolov8_traffic_latest.pt or config.MODEL_NAME)",
    )
    parser.add_argument(
        "--min-map50",
        type=float,
        default=0.60,
        help="Quality Gate: Minimum mAP@0.5 required to promote model to production (default: 0.60)",
    )
    parser.add_argument(
        "--force-deploy",
        action="store_true",
        help="Force promote the new model even if it falls below the Quality Gate threshold",
    )
    parser.add_argument(
        "--auto-deploy",
        action="store_true",
        default=False,
        help="Automatically promote model to production without waiting for UI approval",
    )
    return parser.parse_args()


def get_base_model_path(custom_base: str = None) -> str:
    """Find the best existing model to use as fine-tuning baseline."""
    if custom_base and Path(custom_base).is_file():
        return custom_base

    latest_pt = config.BASE_DIR / "models" / "yolov8_traffic_latest.pt"
    if latest_pt.is_file():
        return str(latest_pt)

    return config.MODEL_NAME


def get_next_version(version_json_path: Path) -> str:
    """Read version.json and calculate the next minor version (e.g. v1.0.0 -> v1.1.0)."""
    if not version_json_path.is_file():
        return "v1.1.0"
    try:
        with open(version_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        current = data.get("active_version", "v1.0.0").lstrip("v")
        major, minor, patch = map(int, current.split("."))
        return f"v{major}.{minor + 1}.0"
    except Exception:
        return f"v1.1.{int(time.time()) % 1000}"


def update_version_registry(
    version_json_path: Path,
    new_version: str,
    model_filename: str,
    metrics: dict,
    samples_count: int,
):
    """Record the newly promoted model into the version history registry."""
    data = {"active_version": new_version, "model_path": f"models/{model_filename}", "history": []}
    if version_json_path.is_file():
        try:
            with open(version_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    data["active_version"] = new_version
    data["model_path"] = f"models/{model_filename}"
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    history_entry = {
        "version": new_version,
        "model_file": model_filename,
        "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": samples_count,
        "metrics": metrics,
        "status": "active",
    }
    data.setdefault("history", []).append(history_entry)

    with open(version_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_arguments()
    dm = DatasetManager()

    print("=" * 70)
    print("🚀 AUTOMATED RETRAINING PIPELINE & QUALITY GATE (STEP 4)")
    print("=" * 70)

    # 1. DATASET INGESTION
    ingestion_manifest = None
    if args.zip:
        zip_path = Path(args.zip)
        print(f"\n📥 Ingesting single dataset archive: {zip_path}")
        dm.extract_and_ingest_zip(zip_path, val_ratio=0.2)
    else:
        # Consolidate all exports in cvat_exports/ or dataset/imports/
        candidate_zips = list(config.CVAT_EXPORTS_DIR.glob("*.zip"))
        if candidate_zips:
            ingestion_manifest = dm.ingest_all_cvat_exports(config.CVAT_EXPORTS_DIR, val_ratio=0.2)
        elif list(dm.imports_dir.glob("*.zip")):
            ingestion_manifest = dm.ingest_all_cvat_exports(dm.imports_dir, val_ratio=0.2)

    stats = dm.get_stats()
    print(f"\n📊 Current Dataset Summary:")
    print(f"   • Training images   : {stats['train_images']}")
    print(f"   • Validation images : {stats['val_images']}")
    print(f"   • Total images      : {stats['total']}")

    if stats["train_images"] == 0:
        print("\n❌ Error: No training images found in dataset/train/images!")
        print("Please export your annotated task from CVAT and run:")
        print("   python retrain.py --zip /path/to/exported_cvat.zip")
        return

    yaml_path = dm.generate_yaml()

    # 2. SELECT BASE MODEL
    base_model_path = get_base_model_path(args.base_model)
    print(f"\n🧠 Base Model for Fine-Tuning : {base_model_path}")
    print(f"⚙️ Target Epochs              : {args.epochs}")
    print(f"⚡ Device                     : {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")

    batch_size = args.batch
    if batch_size == 0:
        batch_size = 16 if torch.cuda.is_available() else 8

    # 3. FINE-TUNING WITH PRODUCTION DATA AUGMENTATION
    print("\n" + "-" * 70)
    print("🔥 Starting Fine-Tuning with Advanced Data Augmentation...")
    print("   ↳ Mosaic: 1.0 (4-angle cross-stitching)")
    print("   ↳ MixUp: 0.15 (blended occlusion)")
    print("   ↳ HSV Augmentation: H=0.015, S=0.7, V=0.4 (camera/weather shifts)")
    print("   ↳ Perspective & Rotation: +/- 10 degrees (camera angle adaptation)")
    print("   ↳ Scale: +/- 50% (multi-distance vehicle scale)")
    print("-" * 70)

    model = YOLO(base_model_path)
    train_results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=batch_size,
        imgsz=config.IMAGE_SIZE,
        lr0=args.lr,
        # --- DATA AUGMENTATION PARAMETERS (FROM check.md) ---
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        scale=0.5,
        perspective=0.0005,
        fliplr=0.5,
        amp=True,
        plots=True,
        verbose=True,
    )

    best_weights_path = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_weights_path.is_file():
        print("❌ Training completed but best.pt was not generated.")
        return

    # 4. AUTOMATED QUALITY GATE & REGRESSION SUITE EVALUATION
    print("\n" + "=" * 70)
    print("🛡️ RUNNING AUTOMATED QUALITY GATE & REGRESSION SUITE")
    print("=" * 70)

    eval_model = YOLO(str(best_weights_path))
    val_results = eval_model.val(data=str(yaml_path), split="val", verbose=False)

    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)

    metrics_dict = {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    print(f"📊 Validation Results on Dataset Splits:")
    print(f"   • mAP@0.5    : {map50:.4f}  (Quality Gate Threshold: {args.min_map50:.2f})")
    print(f"   • mAP@0.5:95 : {map50_95:.4f}")
    print(f"   • Precision  : {precision:.4f}")
    print(f"   • Recall     : {recall:.4f}")

    # Run Golden Test Set Regression Suite
    print("\n" + "-" * 70)
    print("🧪 Evaluating Golden Test Set (Regression Suite)...")
    golden_report = run_regression_test(
        candidate_model_path=best_weights_path,
        baseline_model_path=Path(base_model_path) if Path(base_model_path).is_file() else None,
        min_confidence=config.CONFIDENCE_THRESHOLD,
    )
    if golden_report["total_cases"] > 0:
        print(f"   • Golden Test Cases Passed : {golden_report['passed']}/{golden_report['total_cases']} ({golden_report['pass_rate']}%)")
        for r in golden_report["results"]:
            icon = "✅" if r["status"] == "PASS" else "⚠️"
            print(f"     {icon} Case '{r['case_id']}' [{r['target_class']}]:")
            print(f"        Base Conf: {r['baseline_conf']:.4f} -> Candidate: {r['candidate_conf']:.4f} (Delta: {r['delta_conf']:+.4f}) | Min Req: >={r['required_conf']}")
    else:
        print("   ℹ️ No Golden Test Cases registered yet. (Register in UI Tab 2)")
    print("-" * 70)

    models_dir = config.BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    version_json_path = models_dir / "version.json"

    # Evaluate Quality Gate Condition
    gate_passed = (map50 >= args.min_map50) or args.force_deploy

    if gate_passed:
        staged_dir = stage_model(
            best_weights_path=best_weights_path,
            metrics=metrics_dict,
            samples_count=stats["total"],
            base_model=base_model_path,
            epochs=args.epochs,
            dataset_manifest=ingestion_manifest,
            regression_report=golden_report,
        )

        if args.auto_deploy:
            res = approve_and_deploy_model(str(staged_dir))
            print("\n" + "=" * 70)
            print("🎉 QUALITY GATE PASSED — MODEL AUTO-PROMOTED TO PRODUCTION!")
            print("=" * 70)
            print(f" • New Active Version  : {res['version']}")
            print(f" • Model Artifact      : {res['model_path']}")
            print(f" • Quality Score       : mAP@0.5 = {map50:.4f} (>= {args.min_map50:.2f})")
            print("=" * 70 + "\n")
        else:
            print("\n" + "=" * 70)
            print("📦 QUALITY GATE PASSED — MODEL CANDIDATE STAGED AWAITING APPROVAL")
            print("=" * 70)
            print(f" • Staged Candidate ID : {staged_dir.name}")
            print(f" • Staged Path         : {staged_dir}")
            print(f" • Quality Score       : mAP@0.5 = {map50:.4f} (>= {args.min_map50:.2f})")
            print(f" • Precision           : {precision:.4f} | Recall: {recall:.4f}")
            print("\n👉 ACTION REQUIRED: Open Streamlit Review Studio (Tab '🎯 Model Deployment')")
            print("   to inspect evaluation metrics and APPROVE promotion to production!")
            print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print("⚠️ QUALITY GATE FAILED — DEPLOYMENT REJECTED")
        print("=" * 70)
        print(f" • Observed mAP@0.5    : {map50:.4f} < Required Threshold ({args.min_map50:.2f})")
        print(" • Action Executed     : AUTOMATED ROLLBACK")
        print(f" • Active Model Kept   : {base_model_path}")
        print(" • Note: Candidate weights kept at:", best_weights_path)
        print("   (Use --force-deploy flag if you wish to override and deploy anyway)")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
