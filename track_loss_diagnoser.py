import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 bounding boxes [x1, y1, x2, y2]."""
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
    Tính tỷ lệ diện tích của target_box bị bao phủ/đè lên bởi occluder_box.
    Trả về từ 0.0 (không che) đến 1.0 (bị che phủ 100%).
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
    Chẩn đoán nguyên nhân cốt lõi khiến Candidate bị cảnh báo (Track Loss hoặc Low Confidence).
    Phân loại:
      1. PARTIAL_OCCLUSION -> APPROVE (Mẫu vàng che khuất 1 phần)
      2. DETECTION_DROP    -> APPROVE (YOLO bị sụt áp confidence hoặc bỏ sót hoàn toàn)
      3. FULL_OCCLUSION    -> REJECT  (Bị che khuất 100% sau xe lớn)
      4. ID_SWITCH         -> REJECT  (YOLO vẫn thấy rõ, lỗi do Tracker nhảy ID)
      5. NATURAL_EXIT      -> REJECT  (Xe đi ra khỏi mép khung hình tự nhiên)
    """
    if isinstance(frame_input, (str, os.PathLike)):
        frame = cv2.imread(str(frame_input))
        if frame is None:
            return {"error": f"Không thể đọc ảnh từ: {frame_input}"}
    else:
        frame = frame_input.copy()

    orig_h, orig_w = frame.shape[:2]

    # Lấy thông tin BBox cũ của xe bị mất dấu
    last_bbox = details.get("last_known_bbox") or details.get("bbox")
    target_class = details.get("class_name", "vehicle")
    last_conf = details.get("confidence") or details.get("last_confidence", 0.0)

    if not last_bbox:
        return {
            "verdict": "UNKNOWN",
            "recommendation": "REVIEW",
            "badge_color": "blue",
            "title": "Không đủ tọa độ để chẩn đoán",
            "explanation": "Candidate này không có thông tin last_known_bbox trong metadata.",
        }

    # 1. Kiểm tra xe đi ra mép khung hình (Natural Exit)
    margin = config.BORDER_MARGIN_PIXELS
    x1, y1, x2, y2 = last_bbox
    if x1 <= margin or y1 <= margin or x2 >= (orig_w - margin) or y2 >= (orig_h - margin):
        return {
            "verdict": "NATURAL_EXIT",
            "recommendation": "REJECT",
            "badge_color": "gray",
            "title": "Xe đi ra khỏi mép khung hình (Natural Exit)",
            "explanation": f"Chiếc {target_class} đã chạm sát đường biên mép ảnh ({orig_w}x{orig_h}). Việc mất dấu khi xe rời khung hình là hoàn toàn bình thường.",
            "detected_conf": None,
        }

    # 2. Khởi tạo mô hình và quét với ngưỡng siêu nhạy (conf=0.10)
    if model is None:
        model = YOLO(config.MODEL_NAME)

    # Chạy YOLO với conf thấp để bắt cả các trường hợp sụt confidence
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

            # Kiểm tra độ khớp với vị trí xe cũ
            iou = compute_iou(last_bbox, box_list)
            overlap = compute_overlap_ratio(last_bbox, box_list)

            # Nếu cùng class xe và trùng vị trí (IoU > 0.3)
            if cls_name == target_class and iou >= 0.30:
                matched_same_class.append({
                    "box": box_list,
                    "confidence": float(conf_val),
                    "iou": iou,
                })

            # Kiểm tra xe khác (đặc biệt là xe lớn hơn: car, truck, bus) đè lên vị trí xe này
            if cls_name in ["car", "truck", "bus", "van"] and cls_name != target_class:
                if overlap >= 0.20:
                    occluding_vehicles.append({
                        "class_name": cls_name,
                        "box": box_list,
                        "confidence": float(conf_val),
                        "overlap_ratio": overlap,
                    })

    # Sắp xếp xe che lấp theo mức độ che phủ lớn nhất
    occluding_vehicles.sort(key=lambda x: x["overlap_ratio"], reverse=True)

    # 3. PHÂN TÍCH RA QUYẾT ĐỊNH CHẨN ĐOÁN

    # Kịch bản A: Có xe lớn che lấp
    if occluding_vehicles:
        top_occ = occluding_vehicles[0]
        ratio = top_occ["overlap_ratio"]
        occ_name = top_occ["class_name"]

        # Nếu che gần như 100% (>80% diện tích) -> FULL OCCLUSION
        if ratio >= 0.80:
            return {
                "verdict": "FULL_OCCLUSION",
                "recommendation": "REJECT",
                "badge_color": "gray",
                "title": f"Bị xe {occ_name} che khuất hoàn toàn ({ratio*100:.0f}%)",
                "explanation": f"Chiếc {target_class} đã lọt hẳn vào sau thân xe {occ_name} (che khuất {ratio*100:.0f}%). Mắt người cũng khó thấy xe. Việc này thuộc trách nhiệm của Tracker (tăng track_buffer), không nên ép AI đoán mò.",
                "overlap_ratio": ratio,
                "occluder": occ_name,
            }
        else:
            # Che một phần (20% - 80%) nhưng mắt người vẫn thấy -> PARTIAL OCCLUSION (MẪU VÀNG!)
            return {
                "verdict": "PARTIAL_OCCLUSION",
                "recommendation": "APPROVE",
                "badge_color": "green",
                "title": f"⭐ MẪU VÀNG: Bị {occ_name} che một phần ({ratio*100:.0f}%)",
                "explanation": f"Chiếc {target_class} đang bị xe {occ_name} che khuất {ratio*100:.0f}%, gây ra hiện tượng Feature Occlusion (bị xe lớn nuốt đặc trưng). Mắt người vẫn thấy rõ xe máy! Cần gán nhãn để dạy AI phát hiện xe bị che.",
                "overlap_ratio": ratio,
                "occluder": occ_name,
            }

    # Kịch bản B: YOLO tìm thấy xe ở vị trí cũ
    if matched_same_class:
        best_match = max(matched_same_class, key=lambda x: x["confidence"])
        found_conf = best_match["confidence"]

        # B1: Confidence bị tụt thấp hơn ngưỡng hoạt động
        if found_conf < conf_threshold:
            return {
                "verdict": "DETECTION_DROP",
                "recommendation": "APPROVE",
                "badge_color": "green",
                "title": f"YOLO bị sụt độ tin cậy (Conf: {found_conf:.2f} < {conf_threshold})",
                "explanation": f"YOLO vẫn lờ mờ nhìn thấy chiếc {target_class} nhưng độ tin cậy chỉ đạt {found_conf:.2f} (dưới ngưỡng lọc {conf_threshold}). Cần gán nhãn để AI nâng điểm tự tin ở góc này.",
                "detected_conf": found_conf,
            }
        else:
            # B2: YOLO nhận diện cực tốt (conf >= ngưỡng), nhưng Tracker lại báo mất dấu!
            return {
                "verdict": "ID_SWITCH",
                "recommendation": "REJECT",
                "badge_color": "orange",
                "title": f"Lỗi Tracker nhảy ID (YOLO vẫn bắt tốt với Conf {found_conf:.2f})",
                "explanation": f"Mô hình YOLO hoàn toàn vô tội! Nó vẫn phát hiện xe với độ tin cậy rất cao ({found_conf:.2f}), nhưng thuật toán ByteTrack bị trượt Kalman Filter và gán ID mới. Không cần tốn công gán nhãn ảnh này.",
                "detected_conf": found_conf,
            }

    # Kịch bản C: Quét ở conf=0.10 mà YOLO vẫn hoàn toàn không thấy gì!
    return {
        "verdict": "SEVERE_DETECTION_DROP",
        "recommendation": "APPROVE",
        "badge_color": "green",
        "title": f"YOLO bỏ sót hoàn toàn chiếc {target_class} (False Negative)",
        "explanation": f"Dù hạ ngưỡng phát hiện xuống 0.10, YOLO vẫn hoàn toàn mù tịt trước chiếc {target_class} này. Đây là lỗi nhận diện nghiêm trọng của mô hình, rất cần gán nhãn trên CVAT!",
        "detected_conf": 0.0,
    }
