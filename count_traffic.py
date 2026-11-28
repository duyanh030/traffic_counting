import argparse
from datetime import datetime
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


def write_debug_log(log_path: Path | None, text: str, console: bool = True):
    """Write text to terminal and append timestamped lines to the debug log file."""
    if console:
        print(text)
    if log_path is not None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for line in text.split("\n"):
                    f.write(f"[{now_str}] {line}\n")
        except Exception as e:
            if console:
                print(f"⚠️ Failed to write to debug log file: {e}")


def print_debug_frame_info(frame_idx: int, detections, model, line_y: int, line_zone=None, debug_log_path: Path = None):
    """Print exhaustive per-vehicle tracking, anchor, and LineZone crossing status to terminal and log file."""
    total_dets = len(detections) if (detections is not None and len(detections) > 0) else 0
    in_cnt = line_zone.in_count if line_zone else 0
    out_cnt = line_zone.out_count if line_zone else 0
    anchors_desc = [a.name for a in line_zone.triggering_anchors] if line_zone else ["N/A"]

    buf = []
    buf.append(f"\n{'='*82}")
    buf.append(f"🔍 [DEBUG FRAME #{frame_idx}] INTERACTIVE STEP-BY-STEP INSPECTOR")
    buf.append(f"{'='*82}")
    buf.append("📏 LINEZONE CONFIGURATION:")
    if line_zone:
        buf.append(f"  • Line Vector      : Start ({line_zone.vector.start.x:.0f}, {line_zone.vector.start.y:.0f}) ──► End ({line_zone.vector.end.x:.0f}, {line_zone.vector.end.y:.0f}) | Y = {line_y} px")
        buf.append(f"  • Trigger Anchors  : {', '.join(anchors_desc)} ({len(anchors_desc)} points)")
        buf.append(f"  • Current Counters : Total IN = {in_cnt} | Total OUT = {out_cnt}")
    else:
        buf.append(f"  • Line Y-Position  : {line_y} px")
    buf.append("-" * 82)
    buf.append(f"🚗 ACTIVE VEHICLE TRACKS (Detected: {total_dets}):")
    buf.append("-" * 82)

    if total_dets > 0 and detections.tracker_id is not None:
        # Precompute LineZone anchor states if available
        lz_in_limits = None
        lz_has_left = None
        lz_has_right = None
        if line_zone is not None:
            try:
                lz_in_limits, lz_has_left, lz_has_right = line_zone._compute_anchor_sides(detections)
            except Exception:
                pass

        for idx, (cid, tid, conf, (x1, y1, x2, y2)) in enumerate(zip(
            detections.class_id, detections.tracker_id, detections.confidence, detections.xyxy
        )):
            cname = model.names.get(cid, str(cid))
            tid_int = int(tid)
            w = int(x2 - x1)
            h = int(y2 - y1)
            aspect = round(w / h, 2) if h > 0 else 0.0

            y_top = int(y1)
            y_bot = int(y2)
            y_mid = int((y1 + y2) / 2)
            d_top = y_top - line_y
            d_mid = y_mid - line_y
            d_bot = y_bot - line_y

            # Anchor analysis
            in_lim = bool(lz_in_limits[idx]) if lz_in_limits is not None else True
            has_l = bool(lz_has_left[idx]) if lz_has_left is not None else (y_top < line_y)
            has_r = bool(lz_has_right[idx]) if lz_has_right is not None else (y_bot >= line_y)
            is_straddling = has_l and has_r

            # Determine crossing status
            if line_zone is not None:
                confirmed_side = line_zone._confirmed_crossing_side.get(tid_int)
                conf_side_str = "LEFT (Before Line)" if confirmed_side is True else "RIGHT (Past Line)" if confirmed_side is False else "NOT INITIALIZED"
                history_list = list(line_zone.crossing_state_history.get(tid_int, []))

                if confirmed_side is not None and not is_straddling:
                    current_side = has_l  # True = Left, False = Right
                    if current_side != confirmed_side:
                        cross_status = "✅ CROSSED! (Transitioned across line)"
                    else:
                        cross_status = "STILL ON ORIGINAL SIDE (No crossing yet)"
                elif is_straddling:
                    cross_status = "⚠️ STRADDLING (In Dead-Zone: Head crossed, Tail has not)"
                else:
                    cross_status = "INITIALIZING (First frames on current side)"
            else:
                conf_side_str = "N/A"
                history_list = []
                cross_status = "PAST Line" if y_top >= line_y else "STRADDLING Line" if y_bot >= line_y else "BEFORE Line"

            buf.append(f"[TRACK #{idx+1}] ID: #{tid_int} | Class: {cname} (cid={cid}) | Confidence: {conf:.2f}")
            buf.append(f"  ├─ Bounding Box   : [x1={int(x1)}, y1={int(y1)}, x2={int(x2)}, y2={int(y2)}] | W={w}px, H={h}px (Aspect: {aspect})")
            buf.append(f"  ├─ Y-Dist to Line : Top Edge: {d_top:+d}px | Center: {d_mid:+d}px | Bottom Edge: {d_bot:+d}px (Y_line={line_y})")

            # Detailed anchor points
            buf.append(f"  ├─ Anchors & Sides: (In Limits: {in_lim})")
            if line_zone is not None:
                v_start_x = line_zone.vector.start.x
                v_start_y = line_zone.vector.start.y
                v_dx = line_zone.vector.end.x - v_start_x
                v_dy = line_zone.vector.end.y - v_start_y

                for anchor in line_zone.triggering_anchors:
                    pt = detections[idx:idx+1].get_anchors_coordinates(anchor)[0]
                    cp = v_dx * (pt[1] - v_start_y) - v_dy * (pt[0] - v_start_x)
                    side_name = "LEFT  (Before/Above Line)" if cp < 0 else "RIGHT (Past/Below Line)"
                    buf.append(f"  │   • {anchor.name:<13} at ({pt[0]:.1f}, {pt[1]:.1f}) ──► Side: {side_name} (cp={cp:+.0f})")

            buf.append(f"  ├─ Side Summary   : Has Left Anchors = {has_l} | Has Right Anchors = {has_r}")
            if is_straddling:
                buf.append("  │   ⚠️ DEAD-ZONE ALERT: Vehicle spans both sides! LineZone 'if has_left and has_right: continue' is SKIPPING count updates!")
            buf.append("  └─ LineZone State :")
            buf.append(f"      • Confirmed Origin Side : {conf_side_str}")
            buf.append(f"      • Crossing History Queue: {history_list} (len={len(history_list)}/{line_zone.crossing_history_length if line_zone else '?'})")
            buf.append(f"      • Status Verdict        : {cross_status}\n")
    else:
        buf.append("  (No tracked vehicles detected in this frame)\n")

    buf.append("-" * 82)
    buf.append("👉 Debug Controls: [Space] Tiếp tục / Tạm dừng | [D] Tua 1 Frame tiếp | [Q] Thoát")
    buf.append("=" * 82)

    write_debug_log(debug_log_path, "\n".join(buf), console=True)


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
        start=sv.Point(0, line_y),
        end=sv.Point(width , line_y)
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

    # Interactive Step-by-Step Debug state & file logger
    is_paused = False
    debug_log_file = None
    if args.debug:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        # Format: debug_<source_name>_YYYYMMDD_HHMMSS.log
        debug_log_file = config.LOGS_DIR / f"debug_{source_name}_{start_timestamp}.log"
        init_banner = (
            f"\n{'='*82}\n"
            f"🚀 [DEBUG SESSION INITIALIZED]\n"
            f"• Script Start Time : {start_time_str}\n"
            f"• Video Source      : {source} (Resolution: {width}x{height})\n"
            f"• Model             : {model_name} (Version: {semantic_ver})\n"
            f"• Line Y-Position   : {line_y} px ({line_pos * 100:.1f}% frame height)\n"
            f"• Debug Log File    : {debug_log_file}\n"
            f"• Interactive Keys  :\n"
            f"  - Press [Space]   : Tạm dừng (Pause) / Tiếp tục (Resume)\n"
            f"  - Press [D]       : Tua tới đúng 1 frame tiếp theo\n"
            f"  - Press [Q]       : Thoát chương trình\n"
            f"{'='*82}"
        )
        write_debug_log(debug_log_file, init_banner, console=True)

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
                        print_debug_frame_info(frame_idx, detections, model, line_y, line_zone=line_zone, debug_log_path=debug_log_file)
                        user_quit = False
                        while is_paused:
                            k = cv2.waitKey(0) & 0xFF
                            if k == ord(" "):
                                is_paused = False
                                write_debug_log(debug_log_file, f"▶️ User RESUMED video playback from Frame #{frame_idx}.", console=True)
                                break
                            elif k in (ord("d"), ord("D")):
                                write_debug_log(debug_log_file, f"⏩ User STEPPED 1 frame to Frame #{frame_idx + 1}.", console=True)
                                # Advance exactly 1 frame; keep is_paused = True for next frame
                                break
                            elif k in (ord("q"), ord("Q")):
                                user_quit = True
                                write_debug_log(debug_log_file, f"⏹️ User STOPPED playback at Frame #{frame_idx}.", console=True)
                                break
                        if user_quit:
                            break
                    else:
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord(" "):
                            is_paused = True
                            write_debug_log(debug_log_file, f"\n⏸️ User PAUSED video at Frame #{frame_idx}.", console=True)
                        elif k in (ord("d"), ord("D")):
                            is_paused = True
                            write_debug_log(debug_log_file, f"\n⏸️ User STEPPED into Frame #{frame_idx}.", console=True)
                        elif k in (ord("q"), ord("Q")):
                            write_debug_log(debug_log_file, f"\n⏹️ User STOPPED playback at Frame #{frame_idx}.", console=True)
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

    if args.debug and debug_log_file is not None:
        end_banner = (
            f"\n{'='*82}\n"
            f"⏹️ [DEBUG SESSION COMPLETED]\n"
            f"• Session Start Time : {start_time_str}\n"
            f"• Session End Time   : {end_time_str}\n"
            f"• Total Duration     : {duration_seconds:.2f} seconds | Avg FPS: {avg_fps}\n"
            f"• Total Frames Done  : {frame_idx}\n"
            f"• Vehicles Counted   : IN = {line_zone.in_count} | OUT = {line_zone.out_count}\n"
            f"• Debug Log Saved To : {debug_log_file}\n"
            f"{'='*82}"
        )
        write_debug_log(debug_log_file, end_banner, console=True)

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
