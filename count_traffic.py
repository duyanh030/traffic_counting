import argparse
import os
import re
import time
from pathlib import Path
import cv2
from ultralytics import YOLO
import supervision as sv

import config
from candidate_detector import CandidateDetector

# Optimize RTSP streaming over TCP to prevent packet loss / tearing over Wi-Fi
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def mask_url(url: str) -> str:
    """Mask password in RTSP URL for safe terminal logging."""
    return re.sub(r":([^@]+)@", ":****@", url)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time Traffic Counting & Active Learning Candidate Capture")
    parser.add_argument(
        "--source",
        type=str,
        default=config.RTSP_URL,
        help="Input source: file path to an MP4 video or RTSP URL (default: config.RTSP_URL)",
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
    return parser.parse_args()


def main():
    args = parse_arguments()
    source = args.source
    is_video_file = Path(source).is_file()

    # Capture execution start timestamps
    start_time = time.time()
    start_timestamp = time.strftime("%Y%m%d_%H%M%S")
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Clean model identifier (e.g. "yolov8n" from "yolov8n.pt")
    model_identifier = Path(config.MODEL_NAME).stem

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
    print(f"Initializing YOLO model: {config.MODEL_NAME} (version/id: {model_identifier})")
    print(f"Candidate output folder: {run_candidates_dir}")
    model = YOLO(config.MODEL_NAME)

    # Initialize CandidateDetector with the session-specific folder
    candidate_detector = None
    if args.save_candidates:
        candidate_detector = CandidateDetector(run_candidates_dir=run_candidates_dir)

    # Set horizontal counting line based on configured height percentage
    line_y = int(height * config.COUNTING_LINE_POSITION)
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
            clean_frame = frame.copy()

            # Run YOLO tracking with ByteTrack
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=config.VEHICLE_CLASSES,
                conf=config.CONFIDENCE_THRESHOLD,
                imgsz=config.IMAGE_SIZE,
                verbose=False,
            )[0]

            # Convert detections to Supervision format
            detections = sv.Detections.from_ultralytics(results)

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
                cv2.imshow("Traffic Counting & Active Learning", annotated_frame)
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
    if candidate_detector is not None:
        saved_report = candidate_detector.save_run_report(
            source=source,
            source_name=source_name,
            model_name=config.MODEL_NAME,
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
    print(f" • Model Version / Name : {config.MODEL_NAME}")
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
        print(f" • Candidates Folder    : {run_candidates_dir}")
        print(f" • Session Report JSON  : {run_candidates_dir / 'run_report.json'}")
        print(f" • Global History Log   : {config.REPORTS_DIR / 'runs_history.json'}")
        print(f" • Global History CSV   : {config.REPORTS_DIR / 'runs_history.csv'}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
