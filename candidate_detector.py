import csv
import cv2
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

import config


class CandidateDetector:
    """
    Monitors object detections and tracking states to identify frames
    with potential errors (false negatives, track loss, low confidence).
    
    Includes Production-Grade Filtering:
    1. Visual Deduplication via Perceptual Hashing (dHash)
    2. Image Quality Gates (Blur & Brightness verification)
    3. Global Time-based Rate-Limiting & Run Quota
    """

    def __init__(
        self,
        run_candidates_dir: Optional[Path] = None,
        model_name: str = "yolov8n.pt",
        semantic_version: str = "v1.0.0",
        model_artifact: str = "yolov8n.pt",
        low_conf_range: tuple = (config.LOW_CONF_MIN, config.LOW_CONF_MAX),
        track_loss_min_frames: int = config.TRACK_LOSS_MIN_FRAMES,
        cooldown_frames: int = config.SNAPSHOT_COOLDOWN_FRAMES,
        border_margin: int = config.BORDER_MARGIN_PIXELS,
        max_candidates: int = config.MAX_CANDIDATES_PER_RUN,
        global_cooldown: int = config.GLOBAL_COOLDOWN_FRAMES,
        min_blur_score: float = config.MIN_BLUR_SCORE,
        min_brightness: float = config.MIN_BRIGHTNESS,
        max_brightness: float = config.MAX_BRIGHTNESS,
        hash_max_diff: int = config.HASH_SIMILARITY_MAX_DIFF,
    ):
        self.candidates_dir = Path(run_candidates_dir or config.CANDIDATES_DIR)
        self.img_dir = self.candidates_dir / "images"
        self.preview_dir = self.candidates_dir / "previews"
        self.meta_dir = self.candidates_dir / "metadata"

        # Model provenance metadata
        self.model_name = model_name
        self.semantic_version = semantic_version
        self.model_artifact = model_artifact

        # Ensure output directories exist
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.low_conf_min, self.low_conf_max = low_conf_range
        self.track_loss_min_frames = track_loss_min_frames
        self.cooldown_frames = cooldown_frames
        self.border_margin = border_margin

        # Rate Limiting & Quotas
        self.max_candidates = max_candidates
        self.global_cooldown = global_cooldown
        self.last_global_snapshot_frame: Dict[str, int] = {}

        # Quality Gate Parameters (from check.md)
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        # Perceptual Hashing (dHash) for Visual Deduplication
        self.hash_max_diff = hash_max_diff
        self.captured_hashes: List[int] = []

        # Tracking state cache: tracker_id -> dict
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        # Pending track-loss buffer to avoid false alarms on temporary 1-2 frame dips
        # {tracker_id: {"candidate_frame": frame, "frame_idx": int, "details": dict, "highlight_bbox": bbox, "missing_since": int}}
        self.pending_track_losses: Dict[int, Dict[str, Any]] = {}
        self.track_loss_persistence_frames = 3  # Must be missing for at least 3 consecutive frames
        # Cooldown registry for individual tracker IDs
        self.cooldown_registry: Dict[str, int] = {}

        # Statistics & Metrics
        self.total_candidates_saved = 0
        self.stats_by_reason: Counter = Counter()
        self.stats_by_class: Counter = Counter()

        # Suppression counters
        self.filtered_duplicates_count = 0
        self.filtered_blurry_count = 0
        self.filtered_brightness_count = 0
        self.filtered_quota_reached = False

    @staticmethod
    def compute_dhash(crop: np.ndarray, hash_size: int = 8) -> int:
        """
        Compute 64-bit difference hash (dHash) on vehicle crop
        for ultrafast visual similarity comparison.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return 0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])

    def is_visual_duplicate(self, crop: np.ndarray) -> bool:
        """
        Check if image crop is visually similar to any previously captured candidate.
        Uses Hamming distance on 64-bit perceptual dHash.
        """
        new_hash = self.compute_dhash(crop)
        if new_hash == 0:
            return False

        for existing_hash in self.captured_hashes:
            # bit_count() calculates Hamming distance in O(1)
            dist = (new_hash ^ existing_hash).bit_count()
            if dist <= self.hash_max_diff:
                return True
        return False

    def validate_quality(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Quality Gate: Discard blurry or extreme brightness frames (Quality Gate from check.md).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 1. Blur test (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.min_blur_score:
            return False, "blurry"

        # 2. Brightness test (Mean pixel intensity: 35-225)
        brightness = float(np.mean(gray))
        if not (self.min_brightness <= brightness <= self.max_brightness):
            return False, "extreme_brightness"

        return True, "ok"

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

    def should_capture_object(self, key: str, current_frame_idx: int) -> bool:
        """Debounce capture events for the same specific object tracker."""
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
    ) -> Optional[str]:
        """
        Save raw frame, annotated preview, and metadata JSON after passing all filters.
        """
        # Quota check
        if self.total_candidates_saved >= self.max_candidates:
            if not self.filtered_quota_reached:
                print(f"🛑 [Quota Reached] Maximum {self.max_candidates} diverse candidates captured for this run.")
                self.filtered_quota_reached = True
            return None

        # Global rate-limiting per reason
        last_global = self.last_global_snapshot_frame.get(reason, -999999)
        if frame_idx - last_global < self.global_cooldown:
            return None

        # Image Quality Gate check
        quality_ok, quality_err = self.validate_quality(frame)
        if not quality_ok:
            if quality_err == "blurry":
                self.filtered_blurry_count += 1
            else:
                self.filtered_brightness_count += 1
            return None

        # Visual Deduplication check (dHash on vehicle crop)
        h, w = frame.shape[:2]
        if highlight_bbox is not None:
            x1, y1, x2, y2 = map(int, highlight_bbox)
            crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        else:
            crop = frame

        if self.is_visual_duplicate(crop):
            self.filtered_duplicates_count += 1
            return None

        # Register hash and update global cooldown
        new_hash = self.compute_dhash(crop)
        if new_hash != 0:
            self.captured_hashes.append(new_hash)
        self.last_global_snapshot_frame[reason] = frame_idx

        # Generate unique candidate file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        candidate_id = f"cand_{timestamp}_f{frame_idx:06d}_{reason}"

        img_path = self.img_dir / f"{candidate_id}.jpg"
        preview_path = self.preview_dir / f"{candidate_id}.jpg"
        meta_path = self.meta_dir / f"{candidate_id}.json"

        # 1. Save clean original frame (for CVAT annotation)
        cv2.imwrite(str(img_path), frame)

        # 2. Save annotated preview for UI review
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

        banner_text = f"[{self.semantic_version}] Alert: {reason.upper()} (Frame {frame_idx})"
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
            "model_version": self.semantic_version,
            "model_artifact": self.model_artifact,
            "model_weights": self.model_name,
            "reason": reason,
            "details": details,
            "image_file": f"images/{candidate_id}.jpg",
            "preview_file": f"previews/{candidate_id}.jpg",
            "reviewed": False,
            "status": "pending",
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.total_candidates_saved += 1
        self.stats_by_reason[reason] += 1
        class_name = details.get("class_name", "unknown")
        self.stats_by_class[class_name] += 1

        print(
            f"✨ [Diverse Candidate Captured] #{self.total_candidates_saved} | ID: {candidate_id} | Class: {class_name} | Reason: {reason}"
        )
        return candidate_id

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: Any,
        model_names: Dict[int, str],
    ):
        """
        Analyze current frame detections & tracker states with quality and duplicate filtering.
        """
        if self.total_candidates_saved >= self.max_candidates:
            return

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

                # 1. CHECK LOW CONFIDENCE (Uncertainty Sampling)
                if self.low_conf_min <= conf <= self.low_conf_max:
                    cooldown_key = f"lowconf_id_{tracker_id}"
                    if self.should_capture_object(cooldown_key, frame_idx):
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

        # 2. CHECK TRACK LOSS (False Negative Detection with Persistence Verification)
        # If a track disappears, put it in pending_track_losses first
        for tid, tinfo in self.active_tracks.items():
            if tid not in current_tracked_ids and tinfo["last_frame"] == frame_idx - 1:
                frame_count = tinfo["frame_count"]
                last_bbox = tinfo["last_bbox"]
                if frame_count >= self.track_loss_min_frames and not self.is_near_border(last_bbox, frame.shape):
                    if tid not in self.pending_track_losses:
                        details = {
                            "tracker_id": int(tid),
                            "class_name": tinfo["class_name"],
                            "tracked_duration_frames": frame_count,
                            "last_known_bbox": [float(c) for c in last_bbox],
                            "last_confidence": tinfo["confidence"],
                        }
                        self.pending_track_losses[tid] = {
                            "frame": frame.copy(),
                            "frame_idx": frame_idx,
                            "details": details,
                            "highlight_bbox": last_bbox,
                            "missing_since": frame_idx,
                        }

        # If a pending lost track reappears, cancel the false alarm!
        for tid in list(self.pending_track_losses.keys()):
            if tid in current_tracked_ids:
                del self.pending_track_losses[tid]

        # If a pending lost track remains missing for >= track_loss_persistence_frames, commit snapshot!
        for tid in list(self.pending_track_losses.keys()):
            p = self.pending_track_losses[tid]
            if frame_idx - p["missing_since"] >= self.track_loss_persistence_frames:
                cooldown_key = f"trackloss_id_{tid}"
                if self.should_capture_object(cooldown_key, p["frame_idx"]):
                    self.save_candidate(
                        frame=p["frame"],
                        frame_idx=p["frame_idx"],
                        reason="track_loss",
                        details=p["details"],
                        highlight_bbox=p["highlight_bbox"],
                    )
                del self.pending_track_losses[tid]

        # Cleanup tracks inactive for > 30 frames
        expired_ids = [
            tid
            for tid, tinfo in self.active_tracks.items()
            if frame_idx - tinfo["last_frame"] > 30
        ]
        for tid in expired_ids:
            del self.active_tracks[tid]
            if tid in self.pending_track_losses:
                del self.pending_track_losses[tid]

    def save_run_report(
        self,
        source: str,
        source_name: str,
        model_name: str,
        start_time_str: str,
        end_time_str: str,
        duration_seconds: float,
        total_frames: int,
        vehicles_in: int,
        vehicles_out: int,
        semantic_version: str = "v1.0.0",
    ) -> Dict[str, Any]:
        """
        Generate and save a detailed run report including deduplication & quality metrics.
        """
        clean_source_path = source if not source.startswith("rtsp://") else "rtsp://[camera_stream]"
        fps_avg = round(total_frames / duration_seconds, 2) if duration_seconds > 0 else 0

        report_data = {
            "run_id": self.candidates_dir.name,
            "source_name": source_name,
            "source_path": clean_source_path,
            "model_version": model_name,
            "semantic_version": getattr(self, "semantic_version", semantic_version),
            "model_artifact": getattr(self, "model_artifact", "yolov8n.pt"),
            "model_weights": model_name,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "duration_seconds": round(duration_seconds, 2),
            "total_frames_processed": total_frames,
            "fps_average": fps_avg,
            "counting_summary": {
                "vehicles_in": vehicles_in,
                "vehicles_out": vehicles_out,
                "total_vehicles_counted": vehicles_in + vehicles_out,
            },
            "suspicious_snapshots_summary": {
                "total_captured": self.total_candidates_saved,
                "breakdown_by_reason": dict(self.stats_by_reason),
                "breakdown_by_class": dict(self.stats_by_class),
            },
            "filtering_intelligence": {
                "duplicates_suppressed_by_dhash": self.filtered_duplicates_count,
                "blurry_frames_rejected": self.filtered_blurry_count,
                "extreme_brightness_rejected": self.filtered_brightness_count,
                "max_quota_cap": self.max_candidates,
            },
            "candidates_directory": str(self.candidates_dir),
        }

        # 1. Save specific run_report.json inside this run's folder
        report_path = self.candidates_dir / "run_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 2. Update global reports/runs_history.json
        reports_dir = config.REPORTS_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)
        history_file = reports_dir / "runs_history.json"
        history = []
        if history_file.is_file():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(report_data)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # 3. Update global CSV summary
        csv_file = reports_dir / "runs_history.csv"
        file_exists = csv_file.is_file()
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "run_id",
                    "source_name",
                    "model_version",
                    "semantic_version",
                    "model_artifact",
                    "start_time",
                    "end_time",
                    "duration_sec",
                    "total_frames",
                    "vehicles_in",
                    "vehicles_out",
                    "suspicious_snapshots",
                    "duplicates_filtered",
                    "candidates_dir",
                ])
            writer.writerow([
                report_data["run_id"],
                report_data["source_name"],
                report_data["model_version"],
                report_data["semantic_version"],
                report_data["model_artifact"],
                report_data["start_time"],
                report_data["end_time"],
                report_data["duration_seconds"],
                report_data["total_frames_processed"],
                vehicles_in,
                vehicles_out,
                self.total_candidates_saved,
                self.filtered_duplicates_count,
                str(self.candidates_dir),
            ])

        return report_data
