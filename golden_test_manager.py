import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

import config

GOLDEN_TEST_DIR = config.BASE_DIR / "golden_test_set"
GOLDEN_IMAGES_DIR = GOLDEN_TEST_DIR / "images"
GOLDEN_CASES_JSON = GOLDEN_TEST_DIR / "cases.json"


def ensure_golden_dirs():
    GOLDEN_TEST_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not GOLDEN_CASES_JSON.exists():
        with open(GOLDEN_CASES_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def load_golden_cases() -> List[Dict[str, Any]]:
    ensure_golden_dirs()
    try:
        with open(GOLDEN_CASES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_golden_cases(cases: List[Dict[str, Any]]):
    ensure_golden_dirs()
    with open(GOLDEN_CASES_JSON, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)


def get_golden_case_image_path(case: Dict[str, Any]) -> Optional[Path]:
    """Get absolute Path to the stored golden case image."""
    img_rel = case.get("image_file")
    if not img_rel:
        return None
    p = GOLDEN_TEST_DIR / img_rel
    return p if p.is_file() else None


def draw_golden_case_preview(case: Dict[str, Any]) -> Optional[np.ndarray]:
    """Draw target bounding box and label overlay on golden case image."""
    img_path = get_golden_case_image_path(case)
    if not img_path:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    t_box = case.get("target_bbox")
    if t_box and len(t_box) == 4:
        x1, y1, x2, y2 = map(int, t_box)
        t_cls = case.get("target_class", "target")
        base_conf = case.get("baseline_confidence", 0.0)

        # Draw red bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Label badge
        label = f"[Golden] {t_cls} (Base: {base_conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(th + 8, y1)), (0, 0, 255), -1)
        cv2.putText(
            img,
            label,
            (x1 + 3, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_candidate_to_golden_set(
    meta_path_or_dict: Any,
    image_path: Optional[Path] = None,
    note: str = "",
) -> Dict[str, Any]:
    """
    Save a candidate frame as a permanent regression test case in golden_test_set.
    """
    ensure_golden_dirs()
    if isinstance(meta_path_or_dict, (str, Path)):
        meta_file = Path(meta_path_or_dict)
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if image_path is None:
            # Locate image
            run_dir = meta_file.parent.parent
            img_rel = meta.get("image_file") or f"images/{meta.get('candidate_id')}.jpg"
            image_path = run_dir / img_rel
    else:
        meta = meta_path_or_dict

    cand_id = meta.get("candidate_id") or f"cand_{int(time.time())}"
    details = meta.get("details", {})
    target_box = details.get("last_known_bbox") or details.get("bbox")
    target_class = details.get("class_name", "vehicle")

    if not image_path or not Path(image_path).is_file():
        raise FileNotFoundError(f"Raw candidate image not found at: {image_path}")

    # Copy image to golden_test_set/images/
    dest_image_name = f"{cand_id}.jpg"
    dest_image_path = GOLDEN_IMAGES_DIR / dest_image_name
    shutil.copy2(image_path, dest_image_path)

    # Check baseline confidence with current active production model
    baseline_conf = 0.0
    baseline_ver = "unknown"
    try:
        from model_deployer import get_active_model_info
        info = get_active_model_info()
        baseline_ver = info.get("active_version", "baseline")
        model_path = config.BASE_DIR / info.get("model_path", config.MODEL_NAME)
        if not model_path.is_file():
            model_path = Path(config.MODEL_NAME)
        m = YOLO(str(model_path))
        res = evaluate_single_box(m, dest_image_path, target_box, target_class)
        baseline_conf = res.get("confidence", 0.0)
    except Exception:
        baseline_conf = float(details.get("last_confidence", 0.0) or 0.0)

    golden_entry = {
        "case_id": cand_id,
        "image_file": f"images/{dest_image_name}",
        "frame_idx": meta.get("frame_idx"),
        "target_class": target_class,
        "target_bbox": target_box,  # [x1, y1, x2, y2]
        "baseline_version": baseline_ver,
        "baseline_confidence": round(baseline_conf, 4),
        "min_required_conf": config.CONFIDENCE_THRESHOLD,
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": meta.get("reason", "regression_test"),
        "note": note or f"Captured from frame #{meta.get('frame_idx')} ({meta.get('reason')})",
    }

    cases = load_golden_cases()
    # Replace if case_id already exists, otherwise append
    cases = [c for c in cases if c.get("case_id") != cand_id]
    cases.append(golden_entry)
    save_golden_cases(cases)

    return golden_entry


def evaluate_single_box(
    model: YOLO,
    image_path: Path,
    target_box: List[float],
    target_class: str,
    conf_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Run YOLO on an image and find detection overlapping the target_box.
    """
    if not image_path.is_file():
        return {"detected": False, "confidence": 0.0, "class_name": None, "box": None}

    res = model(str(image_path), conf=conf_threshold, verbose=False)[0]
    best_conf = 0.0
    best_cls = None
    best_box = None

    for b in res.boxes:
        c = float(b.conf[0])
        cls_id = int(b.cls[0])
        cls_name = model.names.get(cls_id, str(cls_id))
        box = b.xyxy[0].tolist()

        # Check overlap: target center inside box or box center inside target
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        tcx = (target_box[0] + target_box[2]) / 2
        tcy = (target_box[1] + target_box[3]) / 2

        inside_target = (target_box[0] <= cx <= target_box[2]) and (target_box[1] <= cy <= target_box[3])
        inside_det = (box[0] <= tcx <= box[2]) and (box[1] <= tcy <= box[3])

        # IoU
        xA = max(target_box[0], box[0])
        yA = max(target_box[1], box[1])
        xB = min(target_box[2], box[2])
        yB = min(target_box[3], box[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        targetArea = max(1.0, (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]))
        overlap = interArea / targetArea

        if inside_target or inside_det or overlap > 0.30:
            if c > best_conf:
                best_conf = c
                best_cls = cls_name
                best_box = box

    return {
        "detected": best_conf >= conf_threshold,
        "confidence": round(best_conf, 4),
        "class_name": best_cls,
        "box": best_box,
    }


def run_regression_test(
    candidate_model_path: Path,
    baseline_model_path: Optional[Path] = None,
    min_confidence: float = config.CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluate all Golden Test Cases against the candidate model (and compare with baseline).
    Returns detailed per-case report and overall pass/fail status.
    """
    cases = load_golden_cases()
    if not cases:
        return {
            "total_cases": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 100.0,
            "results": [],
            "status": "NO_TESTS",
        }

    cand_model = YOLO(str(candidate_model_path))
    base_model = YOLO(str(baseline_model_path)) if baseline_model_path and Path(baseline_model_path).is_file() else None

    results = []
    passed_count = 0

    for case in cases:
        c_id = case.get("case_id")
        img_rel = case.get("image_file")
        img_full = GOLDEN_TEST_DIR / img_rel
        t_box = case.get("target_bbox")
        t_cls = case.get("target_class")
        req_conf = case.get("min_required_conf", min_confidence)
        recorded_base_conf = case.get("baseline_confidence", 0.0)

        # Baseline evaluation
        if base_model:
            base_eval = evaluate_single_box(base_model, img_full, t_box, t_cls)
            active_base_conf = base_eval["confidence"]
        else:
            active_base_conf = recorded_base_conf

        # Candidate evaluation
        cand_eval = evaluate_single_box(cand_model, img_full, t_box, t_cls)
        cand_conf = cand_eval["confidence"]

        is_passed = (cand_conf >= req_conf)
        if is_passed:
            passed_count += 1

        delta_conf = round(cand_conf - active_base_conf, 4)

        results.append({
            "case_id": c_id,
            "frame_idx": case.get("frame_idx"),
            "target_class": t_cls,
            "note": case.get("note"),
            "baseline_conf": active_base_conf,
            "candidate_conf": cand_conf,
            "delta_conf": delta_conf,
            "required_conf": req_conf,
            "status": "PASS" if is_passed else "FAIL",
            "detected_class": cand_eval.get("class_name"),
        })

    pass_rate = round((passed_count / len(cases)) * 100, 1)

    return {
        "total_cases": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "pass_rate": pass_rate,
        "results": results,
        "status": "PASSED" if passed_count == len(cases) else "WARNING",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Golden Test Set & Model Regression Suite")
    parser.add_argument("--candidate", type=str, help="Path to new candidate model weights (.pt)")
    parser.add_argument("--baseline", type=str, default=None, help="Path to current baseline model weights (.pt)")
    parser.add_argument("--add-candidate", type=str, default=None, help="Add a candidate JSON metadata to Golden Test Set")
    parser.add_argument("--note", type=str, default="", help="Note for added golden case")
    parser.add_argument("--list", action="store_true", help="List all Golden Test Cases")
    args = parser.parse_args()

    if args.add_candidate:
        meta_p = Path(args.add_candidate)
        res = add_candidate_to_golden_set(meta_p, note=args.note)
        print("\n" + "=" * 65)
        print("⭐ GOLDEN TEST CASE REGISTERED SUCCESSFULLY!")
        print("=" * 65)
        print(f" • Case ID        : {res['case_id']}")
        print(f" • Target Class   : {res['target_class']}")
        print(f" • Baseline Conf  : {res['baseline_confidence']} (Threshold: {res['min_required_conf']})")
        print(f" • Image Stored   : {res['image_file']}")
        print(f" • Note           : {res['note']}")
        print("=" * 65 + "\n")

    elif args.list:
        cases = load_golden_cases()
        print(f"\n⭐ GOLDEN TEST SET REGISTRY ({len(cases)} cases):")
        for i, c in enumerate(cases, 1):
            print(f" [{i}] {c.get('case_id')} | Class: {c.get('target_class')} | Baseline Conf: {c.get('baseline_confidence')} | Req: {c.get('min_required_conf')}")
            print(f"     ↳ {c.get('note')}")

    elif args.candidate:
        base = args.baseline or (config.BASE_DIR / "models" / "yolov8_traffic_latest.pt")
        print("\n" + "=" * 70)
        print("🧪 RUNNING GOLDEN TEST REGRESSION SUITE")
        print("=" * 70)
        print(f" • Candidate Model : {args.candidate}")
        print(f" • Baseline Model  : {base}")
        report = run_regression_test(Path(args.candidate), Path(base) if Path(base).is_file() else None)
        print(f"\n📊 Summary: {report['passed']}/{report['total_cases']} Cases Passed ({report['pass_rate']}%)")
        print("-" * 70)
        for r in report["results"]:
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(f" {icon} {r['case_id']} [{r['target_class']}]:")
            print(f"     Old Conf: {r['baseline_conf']:.4f} -> New Conf: {r['candidate_conf']:.4f} (Delta: {r['delta_conf']:+.4f}) | Required: >={r['required_conf']}")
        print("=" * 70 + "\n")
    else:
        parser.print_help()
