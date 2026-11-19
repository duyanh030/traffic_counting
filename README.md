# Real-time Traffic Counting & Active Learning System

An automated, real-time traffic detection, tracking, counting, and active learning candidate capture system built with **YOLOv8**, **ByteTrack**, **Roboflow Supervision**, and **OpenCV**, supporting both live **RTSP camera feeds** and **MP4 video files**.

---

## Features

- **Multi-Source Input:** Process live RTSP streams or local video files (`.mp4`, `.avi`, `.mkv`).
- **Real-Time Vehicle Detection:** Powered by Ultralytics YOLOv8 (detects cars, motorcycles, buses, and trucks).
- **Robust Multi-Object Tracking:** Native ByteTrack algorithm integration (`bytetrack.yaml`) for cross-frame ID persistence.
- **Directional Traffic Counting:** Uses Supervision's `LineZone` to count incoming (IN) and outgoing (OUT) vehicles across a virtual boundary.
- **Active Learning Candidate Capture (`CandidateDetector`):**
  - **Track-loss Detection:** Automatically detects and snapshots vehicles that vanish abruptly in the active lane (potential False Negatives).
  - **Uncertainty Sampling:** Captures low-confidence detections (`0.20 <= conf <= 0.38`) for continuous model retraining.
  - **Triple-Output Logging:** Saves clean un-annotated frames (ready for CVAT), highlighted preview images, and metadata JSON files.
- **Secure Configuration:** Sensitive camera credentials (IP, username, password, port) are kept in `.env` and never tracked by Git.

---

## Project Structure

```text
traffic_counting/
├── config.py             # Centralized project configuration module
├── candidate_detector.py # Active Learning: track-loss & uncertainty snapshot engine
├── count_traffic.py      # Main vehicle tracking, counting, and candidate capture
├── test_rtsp.py          # Diagnostic utility to verify RTSP stream connectivity
├── .env.example          # Template for environment variables (committed to git)
├── .env                  # Local credentials and settings (ignored by git)
├── candidates/           # Auto-generated candidate snapshots (ignored by git)
│   ├── images/           # Pristine frames for CVAT annotation
│   ├── previews/         # Highlighted frames for fast human review
│   └── metadata/         # JSON metadata for each candidate
├── .gitignore            # Git ignore rules for caches, weights, venv, candidates
└── README.md             # Project documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/duyanh030/traffic_counting.git
   cd traffic_counting
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install opencv-python ultralytics supervision
   ```

4. **Configure your camera credentials (if using RTSP):**
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

---

## Usage

### 1. Test RTSP Camera Connectivity
```bash
python test_rtsp.py
```

### 2. Run on Live RTSP Stream
```bash
python count_traffic.py
```

### 3. Run on an MP4 Video File
```bash
python count_traffic.py --source path/to/video.mp4
```

### 4. Headless Mode (Server / Batch Mode)
Run without displaying the OpenCV GUI window:
```bash
python count_traffic.py --source demo.mp4 --no-view
```

### 5. Disable Candidate Capturing
```bash
python count_traffic.py --source demo.mp4 --no-candidates
```
- Press **`q`** on the video window to stop early and view the summary in the terminal.

---

## Configuration Reference

Settings can be customized directly in your `.env` file or in [config.py](config.py):
- `RTSP_USERNAME`, `RTSP_PASSWORD`, `RTSP_IP`, `RTSP_PORT`, `RTSP_PATH`: Camera credentials.
- `MODEL_NAME`: YOLO model weights (default: `yolov8n.pt`).
- `CONFIDENCE_THRESHOLD`: Object detection threshold (default: `0.35`).
- `COUNTING_LINE_POSITION`: Vertical position of the counting line (default: `0.6`).
- `LOW_CONF_MIN` / `LOW_CONF_MAX`: Low-confidence boundary range for uncertainty sampling (`0.20 - 0.38`).
- `TRACK_LOSS_MIN_FRAMES`: Minimum frames a track must be active before flagging a sudden loss (`5`).
- `SNAPSHOT_COOLDOWN_FRAMES`: Cooldown between snapshots for the same object (`30`).
