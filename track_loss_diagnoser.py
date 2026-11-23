import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return float(iou)


def compute_overlap_ratio(target_box: List[float], occluder_box: List[float]) -> float:
    """
    Calculate the ratio of target_box area covered/overlapped by occluder_box.
    Returns a float from 0.0 (no overlap) to 1.0 (100% occluded).
    """
    xA = max(target_box[0], occluder_box[0])
    yA = max(target_box[1], occluder_box[1])
    xB = min(target_box[2], occluder_box[2])
    yB = min(target_box[3], occluder_box[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    targetArea = max(1.0, (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]))
    return float(interArea / targetArea)


def diagnose_candidate_frame(
    frame_input: Any,
    details: Dict[str, Any],
    model: Optional[YOLO] = None,
    conf_threshold: float = config.CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Diagnose the root cause of candidate alerts (Track Loss or Low Confidence).
    Classifications:
      1. PARTIAL_OCCLUSION -> APPROVE (Golden Hard Positive: partially occluded)
      2. DETECTION_DROP    -> APPROVE (YOLO confidence dipped or missed)
      3. FULL_OCCLUSION    -> REJECT  (100% hidden behind large vehicle)
      4. ID_SWITCH         -> REJECT  (YOLO detects fine; tracker swapped ID)
      5. NATURAL_EXIT      -> REJECT  (Vehicle naturally exited camera boundary)
    """
    if isinstance(frame_input, (str, os.PathLike)):
        frame = cv2.imread(str(frame_input))
        if frame is None:
            return {"error": f"Unable to read image from: {frame_input}"}
    else:
        frame = frame_input.copy()

    orig_h, orig_w = frame.shape[:2]

    # Extract previous bounding box coordinates of the lost vehicle
    last_bbox = details.get("last_known_bbox") or details.get("bbox")
    target_class = details.get("class_name", "vehicle")
    last_conf = details.get("confidence") or details.get("last_confidence", 0.0)

    if not last_bbox:
        return {
            "verdict": "UNKNOWN",
            "recommendation": "REVIEW",
            "badge_color": "blue",
            "title": "Insufficient coordinates for diagnosis",
            "explanation": "This candidate does not contain last_known_bbox metadata.",
        }

    # 1. Check natural frame exit
    margin = config.BORDER_MARGIN_PIXELS
    x1, y1, x2, y2 = last_bbox
    if x1 <= margin or y1 <= margin or x2 >= (orig_w - margin) or y2 >= (orig_h - margin):
        return {
            "verdict": "NATURAL_EXIT",
            "recommendation": "REJECT",
            "badge_color": "gray",
            "title": "Natural Frame Exit",
            "explanation": f"The {target_class} has touched the camera border margin ({orig_w}x{orig_h}). Tracking termination upon exit is expected.",
            "detected_conf": None,
        }

    # 2. Run low-confidence detection pass (conf=0.10) to capture dipped detections
    if model is None:
        model = YOLO(config.MODEL_NAME)

    results = model(frame, conf=0.10, verbose=False)[0]
    detections = results.boxes

    matched_same_class = []
    occluding_vehicles = []

    if detections is not None and len(detections) > 0:
        boxes_xyxy = detections.xyxy.cpu().numpy()
        confs = detections.conf.cpu().numpy()
        classes = detections.cls.cpu().numpy()
        names = model.names

        for box, conf_val, cls_id in zip(boxes_xyxy, confs, classes):
            cls_name = names.get(int(cls_id), "unknown")
            box_list = [float(c) for c in box]

            iou = compute_iou(last_bbox, box_list)
            overlap = compute_overlap_ratio(last_bbox, box_list)

            # Match with same class at previous location
            if cls_name == target_class and iou >= 0.30:
                matched_same_class.append({
                    "box": box_list,
                    "confidence": float(conf_val),
                    "iou": iou,
                })

            # Check larger vehicles (car, truck, bus, van) overlapping this position
            if cls_name in ["car", "truck", "bus", "van"] and cls_name != target_class:
                if overlap >= 0.20:
                    occluding_vehicles.append({
                        "class_name": cls_name,
                        "box": box_list,
                        "confidence": float(conf_val),
                        "overlap_ratio": overlap,
                    })

    occluding_vehicles.sort(key=lambda x: x["overlap_ratio"], reverse=True)

    # 3. DIAGNOSTIC DECISION LOGIC

    # Scenario A: Occluded by another larger vehicle
    if occluding_vehicles:
        top_occ = occluding_vehicles[0]
        ratio = top_occ["overlap_ratio"]
        occ_name = top_occ["class_name"]

        # If almost 100% occluded (>80%) -> FULL OCCLUSION
        if ratio >= 0.80:
            return {
                "verdict": "FULL_OCCLUSION",
                "recommendation": "REJECT",
                "badge_color": "gray",
                "title": f"Fully Occluded by {occ_name.title()} ({ratio*100:.0f}%)",
                "explanation": f"The {target_class} is almost completely obscured behind a {occ_name} ({ratio*100:.0f}% coverage). The vehicle is not visible to humans either. Let ByteTrack handle this via track_buffer; avoid forcing the detector to hallucinate.",
                "overlap_ratio": ratio,
                "occluder": occ_name,
            }
        else:
            # Partially occluded (20% - 80%) but still visible -> PARTIAL OCCLUSION (GOLDEN EXAMPLE!)
            return {
                "verdict": "PARTIAL_OCCLUSION",
                "recommendation": "APPROVE",
                "badge_color": "green",
                "title": f"⭐ GOLDEN SAMPLE: Partially Occluded by {occ_name.title()} ({ratio*100:.0f}%)",
                "explanation": f"The {target_class} is {ratio*100:.0f}% covered by a {occ_name}, causing Feature Occlusion. The vehicle is still clearly visible to humans! Annotating this sample teaches YOLO to detect partially occluded vehicles.",
                "overlap_ratio": ratio,
                "occluder": occ_name,
            }

    # Scenario B: YOLO finds the vehicle at previous position
    if matched_same_class:
        best_match = max(matched_same_class, key=lambda x: x["confidence"])
        found_conf = best_match["confidence"]

        # B1: Confidence dipped below threshold
        if found_conf < conf_threshold:
            return {
                "verdict": "DETECTION_DROP",
                "recommendation": "APPROVE",
                "badge_color": "green",
                "title": f"YOLO Confidence Dipped (Conf: {found_conf:.2f} < {conf_threshold})",
                "explanation": f"YOLO faintly detects the {target_class} but confidence dropped to {found_conf:.2f} (below threshold {conf_threshold}). Labeling this frame will bolster model confidence under this lighting/angle.",
                "detected_conf": found_conf,
            }
        else:
            # B2: YOLO detected vehicle strongly, but tracker dropped/swapped ID
            return {
                "verdict": "ID_SWITCH",
                "recommendation": "REJECT",
                "badge_color": "orange",
                "title": f"Tracker ID Switch (YOLO detected with Conf: {found_conf:.2f})",
                "explanation": f"YOLO detector is functioning properly with high confidence ({found_conf:.2f}). ByteTrack experienced a Kalman Filter tracking discontinuity and assigned a new ID. No need to annotate this frame.",
                "detected_conf": found_conf,
            }

    # Scenario C: Completely missed even at conf=0.10
    return {
        "verdict": "SEVERE_DETECTION_DROP",
        "recommendation": "APPROVE",
        "badge_color": "green",
        "title": f"YOLO Missed {target_class.title()} (False Negative)",
        "explanation": f"Even with threshold lowered to 0.10, YOLO completely failed to detect this {target_class}. This is a genuine detector false negative and should be annotated in CVAT.",
        "detected_conf": 0.0,
    }
