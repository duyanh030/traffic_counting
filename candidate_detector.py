import cv2
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

import config


class CandidateDetector:
    """
    Monitors object detections and tracking states to identify frames
    with potential errors (false negatives, track loss, low confidence)
    for human review and active learning.
    """

    def __init__(
        self,
        candidates_dir: Optional[Path] = None,
        low_conf_range: tuple = (config.LOW_CONF_MIN, config.LOW_CONF_MAX),
        track_loss_min_frames: int = config.TRACK_LOSS_MIN_FRAMES,
        cooldown_frames: int = config.SNAPSHOT_COOLDOWN_FRAMES,
        border_margin: int = config.BORDER_MARGIN_PIXELS,
    ):
        self.candidates_dir = Path(candidates_dir or config.CANDIDATES_DIR)
        self.img_dir = self.candidates_dir / "images"
        self.preview_dir = self.candidates_dir / "previews"
        self.meta_dir = self.candidates_dir / "metadata"

        # Ensure output directories exist
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.low_conf_min, self.low_conf_max = low_conf_range
        self.track_loss_min_frames = track_loss_min_frames
        self.cooldown_frames = cooldown_frames
        self.border_margin = border_margin

        # Tracking state cache: tracker_id -> dict
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        # Cooldown registry to avoid spamming snapshots for the same object
        self.cooldown_registry: Dict[str, int] = {}
        self.total_candidates_saved = 0

    def is_near_border(self, bbox: np.ndarray, frame_shape: tuple) -> bool:
        """Check if bounding box [x1, y1, x2, y2] is near frame borders (natural exit)."""
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        return (
            x1 <= self.border_margin
            or y1 <= self.border_margin
            or x2 >= (width - self.border_margin)
            or y2 >= (height - self.border_margin)
        )

    def should_capture(self, key: str, current_frame_idx: int) -> bool:
        """Debounce capture events to avoid redundant consecutive snapshots."""
        last_frame = self.cooldown_registry.get(key, -999999)
        if current_frame_idx - last_frame >= self.cooldown_frames:
            self.cooldown_registry[key] = current_frame_idx
            return True
        return False

    def save_candidate(
        self,
        frame: np.ndarray,
        frame_idx: int,
        reason: str,
        details: Dict[str, Any],
        highlight_bbox: Optional[np.ndarray] = None,
    ) -> str:
        """
        Save raw frame, annotated preview, and metadata JSON for a candidate.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        candidate_id = f"cand_{timestamp}_f{frame_idx:06d}_{reason}"

        img_path = self.img_dir / f"{candidate_id}.jpg"
        preview_path = self.preview_dir / f"{candidate_id}.jpg"
        meta_path = self.meta_dir / f"{candidate_id}.json"

        # 1. Save clean original frame (suitable for CVAT annotation)
        cv2.imwrite(str(img_path), frame)

        # 2. Save annotated preview for fast human inspection in the UI
        preview = frame.copy()
        if highlight_bbox is not None:
            x1, y1, x2, y2 = map(int, highlight_bbox)
            color = (0, 0, 255) if reason == "track_loss" else (0, 165, 255)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
            label = f"[{reason}] {details.get('class_name', '')} {details.get('confidence', 0):.2f}"
            cv2.putText(
                preview,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        # Banner at the top
        banner_text = f"Candidate Alert: {reason.upper()} (Frame {frame_idx})"
        cv2.putText(
            preview,
            banner_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(preview_path), preview)

        # 3. Save metadata JSON
        metadata = {
            "candidate_id": candidate_id,
            "frame_idx": frame_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "details": details,
            "image_file": f"images/{candidate_id}.jpg",
            "preview_file": f"previews/{candidate_id}.jpg",
            "reviewed": False,
            "status": "pending",  # pending, approved, rejected
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.total_candidates_saved += 1
        print(f"⚠️ [Candidate Captured] ID: {candidate_id} | Reason: {reason} | Total: {self.total_candidates_saved}")
        return candidate_id

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: Any,
        model_names: Dict[int, str],
    ):
        """
        Analyze current frame detections & tracker states:
        1. Flag low-confidence detections.
        2. Detect track-loss (objects that suddenly vanish in the ROI).
        """
        current_tracked_ids = set()

        if detections is not None and len(detections) > 0 and detections.tracker_id is not None:
            for bbox, conf, class_id, tracker_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
                detections.tracker_id,
            ):
                current_tracked_ids.add(tracker_id)
                class_name = model_names.get(int(class_id), f"class_{class_id}")

                # Update track history
                if tracker_id not in self.active_tracks:
                    self.active_tracks[tracker_id] = {
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "frame_count": 1,
                        "last_bbox": bbox,
                        "class_name": class_name,
                        "confidence": float(conf),
                    }
                else:
                    self.active_tracks[tracker_id]["last_frame"] = frame_idx
                    self.active_tracks[tracker_id]["frame_count"] += 1
                    self.active_tracks[tracker_id]["last_bbox"] = bbox
                    self.active_tracks[tracker_id]["confidence"] = float(conf)

                # 1. CHECK LOW CONFIDENCE (Uncertainty sampling)
                if self.low_conf_min <= conf <= self.low_conf_max:
                    cooldown_key = f"lowconf_id_{tracker_id}"
                    if self.should_capture(cooldown_key, frame_idx):
                        details = {
                            "tracker_id": int(tracker_id),
                            "class_name": class_name,
                            "confidence": float(conf),
                            "bbox": [float(c) for c in bbox],
                        }
                        self.save_candidate(
                            frame=frame,
                            frame_idx=frame_idx,
                            reason="low_confidence",
                            details=details,
                            highlight_bbox=bbox,
                        )

        # 2. CHECK TRACK LOSS (False Negative detection)
        # Identify tracks that were active on the previous frame but are absent now
        missing_ids = [
            tid
            for tid, tinfo in self.active_tracks.items()
            if tid not in current_tracked_ids and tinfo["last_frame"] == frame_idx - 1
        ]

        for tid in missing_ids:
            tinfo = self.active_tracks[tid]
            frame_count = tinfo["frame_count"]
            last_bbox = tinfo["last_bbox"]

            # Only consider objects that were tracked consistently for at least N frames
            if frame_count >= self.track_loss_min_frames:
                # If the object was NOT near the border when it disappeared -> Sudden track loss!
                if not self.is_near_border(last_bbox, frame.shape):
                    cooldown_key = f"trackloss_id_{tid}"
                    if self.should_capture(cooldown_key, frame_idx):
                        details = {
                            "tracker_id": int(tid),
                            "class_name": tinfo["class_name"],
                            "tracked_duration_frames": frame_count,
                            "last_known_bbox": [float(c) for c in last_bbox],
                            "last_confidence": tinfo["confidence"],
                        }
                        self.save_candidate(
                            frame=frame,
                            frame_idx=frame_idx,
                            reason="track_loss",
                            details=details,
                            highlight_bbox=last_bbox,
                        )

        # Cleanup tracks that have been inactive for more than 30 frames
        expired_ids = [
            tid
            for tid, tinfo in self.active_tracks.items()
            if frame_idx - tinfo["last_frame"] > 30
        ]
        for tid in expired_ids:
            del self.active_tracks[tid]
