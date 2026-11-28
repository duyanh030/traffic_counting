import argparse
import os
import re
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

import config
from candidate_detector import CandidateDetector
from model_deployer import get_active_model_info

# Optimize RTSP streaming over TCP to prevent packet loss / tearing over Wi-Fi
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def mask_url(url: str) -> str:
    """Mask password in RTSP URL for safe terminal logging."""
    return re.sub(r":([^@]+)@", ":****@", url)


def print_debug_frame_info(frame_idx: int, detections, model, line_y: int):
    """Print detailed per-vehicle tracking and crossing status to terminal when paused."""
    print(f"\n{'='*28} [DEBUG FRAME #{frame_idx}] {'='*28}")
    print(f"📍 Line Y-Coordinate: {line_y} px | Active Tracked Vehicles: {len(detections) if detections is not None else 0}")
    print("-" * 72)
    if detections is not None and len(detections) > 0 and detections.tracker_id is not None:
        for cid, tid, conf, (x1, y1, x2, y2) in zip(
            detections.class_id, detections.tracker_id, detections.confidence, detections.xyxy
        ):
            cname = model.names.get(cid, str(cid))
            y_top = int(y1)
            y_bot = int(y2)
            delta_bot = y_bot - line_y
            delta_top = y_top - line_y

            if y_bot < line_y:
                pos_status = f"BEFORE Line ({delta_bot} px)"
                crossed = "NO"
            elif y_top <= line_y <= y_bot:
                pos_status = f"STRADDLING Line [Dead-Zone] (Top: {delta_top}px | Bot: +{delta_bot}px)"
                crossed = "STRADDLE"
            else:
                pos_status = f"PAST Line (+{delta_top} px)"
                crossed = "YES"

            print(
                f"[Frame {frame_idx}] ID #{tid:<3} ({cname:<10}) | BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] | "
                f"Conf: {conf:.2f} | Y-dist to Line: {delta_bot:+d}px | Status: {pos_status} | Crossed: {crossed}"
            )
    else:
        print("  (No tracked vehicles detected in this frame)")
    print("-" * 72)
    print("👉 Debug Controls: [Space] Tiếp tục / Tạm dừng | [D] Tua 1 Frame tiếp | [Q] Thoát")
    print("=" * 72)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time Traffic Counting & Active Learning Candidate Capture")
    parser.add_argument(
        "--source",
        type=str,
        default=config.RTSP_URL,
        help="Input source: file path to an MP4 video or RTSP URL (default: config.RTSP_URL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help=f"Path or name of YOLO model weights (default: {config.MODEL_NAME})",
    )
    parser.add_argument(
        "--save-candidates",
        action="store_true",
        default=True,
        help="Automatically capture suspicious frames (low confidence, track loss) for active learning (default: True)",
    )
    parser.add_argument(
        "--no-candidates",
        dest="save_candidates",
        action="store_false",
        help="Disable automatic candidate snapshot capture",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Run in headless mode without displaying the GUI window (useful for servers or batch video processing)",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=config.COUNTING_LINE_POSITION,
        help=f"Vertical position of the counting line as a fraction of frame height (0.0=top, 0.5=center, 1.0=bottom, default: {config.COUNTING_LINE_POSITION})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit number of frames to process for quick benchmark/testing (0 = full video/stream)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable interactive Step-by-Step debug mode (Space: Pause/Resume, D: Step 1 frame, with terminal inspector)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    source = args.source
    is_video_file = Path(source).is_file()

    # Capture execution start timestamps
    start_time = time.time()
    start_timestamp = time.strftime("%Y%m%d_%H%M%S")
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Clean model identifier (e.g. "yolov8_traffic_latest" or "yolov8n")
    model_name = args.model
    model_identifier = Path(model_name).stem

    # Clean source identifier (e.g. "traffic_video" from "traffic_video.mp4" or "rtsp_camera")
    source_name = Path(source).stem if is_video_file else "rtsp_camera"

    # Construct session-specific candidates folder: candidates/<model_identifier>/run_<source_name>_<start_timestamp>
    run_id = f"run_{source_name}_{start_timestamp}"
    run_candidates_dir = config.CANDIDATES_DIR / model_identifier / run_id

    if is_video_file:
        print(f"🎬 Processing video file: {source}")
        cap = cv2.VideoCapture(source)
    else:
        print(f"📡 Connecting to live camera stream: {mask_url(source)}")
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Error: Unable to open video source: {mask_url(source)}")
        return

    # Read first frame to determine video dimensions
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to read the first frame from the source.")
        return

    height, width, _ = frame.shape
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video_file else 0

    print(f"Resolution: {width}x{height} | Total video frames: {total_video_frames or 'Live Stream'}")
    print(f"Initializing YOLO model: {model_name} (version/id: {model_identifier})")
    print(f"Candidate output folder: {run_candidates_dir}")
    model = YOLO(model_name)
    vehicle_class_ids = config.get_vehicle_class_ids(model)
    active_class_names = [model.names[i] for i in vehicle_class_ids]
    print(f"Target vehicle class IDs: {vehicle_class_ids} ({', '.join(active_class_names)})")

    active_info = get_active_model_info()
    semantic_ver = active_info.get("active_version", "v1.0.0")
    model_artifact = Path(active_info.get("model_path", model_name)).name
    print(f"Active Model Semantic Version: {semantic_ver} (Artifact: {model_artifact})")

    # Initialize CandidateDetector with the session-specific folder and model lineage
    candidate_detector = None
    if args.save_candidates:
        candidate_detector = CandidateDetector(
            run_candidates_dir=run_candidates_dir,
            model_name=model_name,
            semantic_version=semantic_ver,
            model_artifact=model_artifact,
        )

    # Set horizontal counting line based on configured height percentage
    line_pos = args.line_position
    line_y = int(height * line_pos)
    print(f"Counting line position: y={line_y} ({line_pos * 100:.1f}% frame height)")
    line_zone = sv.LineZone(
        start=sv.Point(30, line_y),
        end=sv.Point(width - 30, line_y)
    )

    # Initialize annotation tools
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    line_annotator = sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=0.8)

    print(">>> System is running! Press 'q' on the video window to stop.")
    frame_idx = 0

    # Rewind to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize custom ByteTrack config path
    custom_tracker_cfg = config.BASE_DIR / "bytetrack.yaml"
    tracker_config_arg = str(custom_tracker_cfg) if custom_tracker_cfg.is_file() else "bytetrack.yaml"
    print(f"Tracking engine: ByteTrack (Config: {tracker_config_arg}, Agnostic NMS: Enabled)")

    # Dictionary to maintain smoothed/majority class for each vehicle track
    track_class_history = {}  # {tracker_id: {class_id: count}}

    # Interactive Step-by-Step Debug state
    is_paused = False
    if args.debug:
        print("\n🔧 [DEBUG MODE ACTIVATED] Step-by-Step Frame Inspection:")
        print("   • Press [Space] : Tạm dừng (Pause) / Tiếp tục (Resume)")
        print("   • Press [D]     : Tua tới đúng 1 frame tiếp theo")
        print("   • Press [Q]     : Thoát chương trình\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    print("\n✅ Video playback completed.")
                    break
                else:
                    print("Waiting for camera feed...")
                    continue

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                print(f"\n⏱️ Reached max frames limit ({args.max_frames}). Stopping test run.")
                break
            clean_frame = frame.copy()

            # Run YOLO tracking with ByteTrack and Agnostic NMS
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_config_arg,
                classes=vehicle_class_ids,
                conf=config.CONFIDENCE_THRESHOLD,
                imgsz=config.IMAGE_SIZE,
                agnostic_nms=True,
                verbose=False,
            )[0]

            # Convert detections to Supervision format
            detections = sv.Detections.from_ultralytics(results)

            # Smooth class prediction across track lifetime (Majority Voting)
            if detections.tracker_id is not None and len(detections) > 0:
                smoothed_class_ids = []
                for cid, tid in zip(detections.class_id, detections.tracker_id):
                    hist = track_class_history.setdefault(tid, {})
                    hist[cid] = hist.get(cid, 0) + 1
                    # Pick class with majority occurrence
                    majority_cid = max(hist, key=hist.get)
                    smoothed_class_ids.append(majority_cid)
                detections.class_id = np.array(smoothed_class_ids, dtype=int)

            # Active Learning: Detect potential errors (track loss, low confidence)
            if candidate_detector is not None:
                candidate_detector.process_frame(
                    frame=clean_frame,
                    frame_idx=frame_idx,
                    detections=detections,
                    model_names=model.names,
                )

            # Process vehicle tracking and line counting
            if detections.tracker_id is not None and len(detections) > 0:
                # 1. Trigger counting line
                line_zone.trigger(detections)

                # 2. Format labels with tracking IDs and class names
                labels = [
                    f"#{tracker_id} {model.names[class_id]} {conf:.2f}"
                    for class_id, tracker_id, conf in zip(
                        detections.class_id, detections.tracker_id, detections.confidence
                    )
                ]

                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            else:
                annotated_frame = frame.copy()

            # Draw counting line and IN/OUT counters
            line_annotator.annotate(frame=annotated_frame, line_counter=line_zone)

            # Display window (unless running in headless mode)
            if not args.no_view:
                if args.debug:
                    status_lbl = f"[DEBUG PAUSED] Frame #{frame_idx}" if is_paused else f"[DEBUG ACTIVE] Frame #{frame_idx}"
                    lbl_color = (0, 0, 255) if is_paused else (0, 255, 0)
                    cv2.putText(
                        annotated_frame,
                        f"{status_lbl} | Space: Pause/Resume | D: Next Frame | Q: Quit",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        lbl_color,
                        2,
                    )

                cv2.imshow("Traffic Counting & Active Learning", annotated_frame)

                if args.debug:
                    if is_paused:
                        print_debug_frame_info(frame_idx, detections, model, line_y)
                        user_quit = False
                        while is_paused:
                            k = cv2.waitKey(0) & 0xFF
                            if k == ord(" "):
                                is_paused = False
                                print("▶️ Resumed video playback.")
                                break
                            elif k in (ord("d"), ord("D")):
                                # Advance exactly 1 frame; keep is_paused = True for next frame
                                break
                            elif k in (ord("q"), ord("Q")):
                                user_quit = True
                                break
                        if user_quit:
                            print("\n⏹️ Stopped by user.")
                            break
                    else:
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord(" "):
                            is_paused = True
                            print(f"\n⏸️ Paused at Frame #{frame_idx}.")
                        elif k in (ord("d"), ord("D")):
                            is_paused = True
                            print(f"\n⏸️ Stepping at Frame #{frame_idx}.")
                        elif k in (ord("q"), ord("Q")):
                            print("\n⏹️ Stopped by user.")
                            break
                else:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n⏹️ Stopped by user.")
                        break
    finally:
        cap.release()
        if not args.no_view:
            cv2.destroyAllWindows()

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    duration_seconds = end_time - start_time
    avg_fps = round(frame_idx / duration_seconds, 2) if duration_seconds > 0 else 0

    # Save detailed run report
    saved_report = None
    active_info = get_active_model_info()
    semantic_ver = active_info.get("active_version", "v1.0.0")

    if candidate_detector is not None:
        saved_report = candidate_detector.save_run_report(
            source=source,
            source_name=source_name,
            model_name=model_name,
            semantic_version=semantic_ver,
            start_time_str=start_time_str,
            end_time_str=end_time_str,
            duration_seconds=duration_seconds,
            total_frames=frame_idx,
            vehicles_in=line_zone.in_count,
            vehicles_out=line_zone.out_count,
        )

    # Display final comprehensive summary
    print("\n" + "=" * 70)
    print("📊 FINAL EXECUTION REPORT")
    print("=" * 70)
    print(f" • Run Session ID       : {run_id}")
    print(f" • Video Source Name    : {source_name}")
    print(f" • Video Source Path    : {mask_url(source) if not is_video_file else source}")
    print(f" • Model Weights Path   : {model_name}")
    print(f" • Active Model Version : {semantic_ver}")
    print(f" • Start Time           : {start_time_str}")
    print(f" • End Time             : {end_time_str}")
    print(f" • Duration             : {duration_seconds:.2f} seconds (~{duration_seconds/60:.1f} mins) | Avg FPS: {avg_fps}")
    print(f" • Total Frames Done    : {frame_idx}")
    print(f" • Vehicles Counted IN  : {line_zone.in_count}")
    print(f" • Vehicles Counted OUT : {line_zone.out_count}")
    print(f" • Total Vehicles       : {line_zone.in_count + line_zone.out_count}")

    if candidate_detector is not None:
        print("-" * 70)
        print(f" • Suspicious Snapshots : {candidate_detector.total_candidates_saved}")
        if candidate_detector.stats_by_reason:
            print(f"   ↳ By Reason          : {dict(candidate_detector.stats_by_reason)}")
        if candidate_detector.stats_by_class:
            print(f"   ↳ By Class           : {dict(candidate_detector.stats_by_class)}")
        if candidate_detector.filtered_duplicates_count or candidate_detector.filtered_blurry_count or candidate_detector.filtered_brightness_count:
            print(f" • Deduplication (dHash): {candidate_detector.filtered_duplicates_count} redundant frames suppressed")
            print(f" • Quality Filtering   : {candidate_detector.filtered_blurry_count} blurry, {candidate_detector.filtered_brightness_count} extreme brightness rejected")
        print(f" • Candidates Folder    : {run_candidates_dir}")
        print(f" • Session Report JSON  : {run_candidates_dir / 'run_report.json'}")
        print(f" • Global History Log   : {config.REPORTS_DIR / 'runs_history.json'}")
        print(f" • Global History CSV   : {config.REPORTS_DIR / 'runs_history.csv'}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
