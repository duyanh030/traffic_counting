import cv2
import os
import re
from ultralytics import YOLO
import supervision as sv

import config

# Optimize RTSP streaming over TCP to prevent packet loss / tearing over Wi-Fi
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def mask_url(url: str) -> str:
    """Mask password in RTSP URL for safe terminal logging."""
    return re.sub(r":([^@]+)@", ":****@", url)


def main():
    print(f"Connecting to camera: {mask_url(config.RTSP_URL)}")
    print(f"Initializing model '{config.MODEL_NAME}' with Native ByteTrack...")
    model = YOLO(config.MODEL_NAME)

    cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Unable to connect to RTSP stream: {mask_url(config.RTSP_URL)}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read the first frame from the camera stream.")
        return

    height, width, _ = frame.shape

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Waiting for camera feed...")
            continue

        # Use built-in YOLO tracking (ByteTrack)
        # persist=True maintains track IDs across consecutive frames
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=config.VEHICLE_CLASSES,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.IMAGE_SIZE,
            verbose=False
        )[0]

        # Convert results to Supervision format
        detections = sv.Detections.from_ultralytics(results)

        # Process when tracked objects are detected
        if detections.tracker_id is not None:
            # 1. Trigger counting line
            line_zone.trigger(detections)

            # 2. Format labels with tracking IDs and class names
            labels = [
                f"#{tracker_id} {model.names[class_id]}"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
            ]

            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        else:
            annotated_frame = frame.copy()

        # Draw counting line and IN/OUT counters
        line_annotator.annotate(frame=annotated_frame, line_counter=line_zone)

        # Display window
        cv2.imshow("Traffic Counting System - RTSP Camera", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n--- SUMMARY: Vehicles IN: {line_zone.in_count} | Vehicles OUT: {line_zone.out_count} ---")


if __name__ == "__main__":
    main()
