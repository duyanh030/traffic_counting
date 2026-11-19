# Real-time Traffic Counting & Active Learning System

An automated, end-to-end traffic detection, tracking, counting, and active learning candidate capture system built with **YOLOv8**, **ByteTrack**, **Roboflow Supervision**, **Streamlit**, and **OpenCV**, supporting both live **RTSP camera feeds** and **MP4 video files**.

---

## 🚀 Active Learning & MLOps Architecture

```text
┌──────────────────────┐
│  Input Video / RTSP  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  count_traffic.py (YOLOv8 + ByteTrack)       │ ──► Vehicles IN / OUT Counting
│  + CandidateDetector Engine                  │
└──────────┬───────────────────────────────────┘
           │ (Auto-snapshots False Negatives & Low-Conf)
           ▼
┌──────────────────────────────────────────────┐
│  candidates/<model>/run_<source>_<timestamp> │ ──► Clean Images + Highlight Previews + JSON
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  app.py (Streamlit Review Studio)            │ ──► Human-in-the-loop: Approve / Reject
└──────────┬───────────────────────────────────┘
           │ (Export Approved Samples)
           ▼
┌──────────────────────────────────────────────┐
│  CVAT Annotation & Retraining Pipeline       │ ──► Fine-tune -> Benchmark Gate -> Deploy
└──────────────────────────────────────────────┘
```

---

## Key Features

- **Multi-Source Input:** Process live RTSP streams or local video files (`.mp4`, `.avi`, `.mkv`).
- **Real-Time Vehicle Detection:** Powered by Ultralytics YOLOv8 (detects cars, motorcycles, buses, and trucks).
- **Robust Multi-Object Tracking:** Native ByteTrack algorithm integration (`bytetrack.yaml`) for cross-frame ID persistence.
- **Directional Traffic Counting:** Uses Supervision's `LineZone` to count incoming (IN) and outgoing (OUT) vehicles across a virtual boundary.
- **Active Learning Candidate Capture (`CandidateDetector`):**
  - **Track-loss Detection:** Automatically detects and snapshots vehicles that vanish abruptly in the active lane (potential False Negatives).
  - **Uncertainty Sampling:** Captures low-confidence detections (`0.20 <= conf <= 0.38`) for continuous model retraining.
  - **Triple-Output Logging:** Saves clean un-annotated frames (ready for CVAT), highlighted preview images, and metadata JSON files.
- **Interactive Review Studio (Streamlit Web UI):**
  - Browse candidate snapshots grouped by model version and session timestamp.
  - Filter by review status (Pending / Approved / Rejected), detection reason, and vehicle class.
  - Single-click or batch approval/rejection to prepare datasets for CVAT annotation.
  - Side-by-side inspection of pristine raw frames vs. AI-highlighted preview frames.
- **Partitioned Execution Sessions & Reports:**
  - Candidates are organized into distinct folders by **model version** and **timestamped run session** (`candidates/<model>/run_<source>_<timestamp>/`).
  - Automated generation of session-specific `run_report.json` and global history logs (`reports/runs_history.json` and `reports/runs_history.csv`).
- **Secure Configuration:** Sensitive camera credentials (IP, username, password, port) are kept in `.env` and never tracked by Git.

---

## Project Structure

```text
traffic_counting/
├── config.py             # Centralized project configuration module
├── candidate_detector.py # Active Learning: track-loss, uncertainty snapshot & reporting engine
├── count_traffic.py      # Main vehicle tracking, counting, and candidate capture
├── app.py                # Streamlit Web Studio for candidate review and approval
├── test_rtsp.py          # Diagnostic utility to verify RTSP stream connectivity
├── .env.example          # Template for environment variables (committed to git)
├── .env                  # Local credentials and settings (ignored by git)
├── candidates/           # Auto-generated candidate snapshots per run (ignored by git)
│   └── <model_version>/  # e.g., yolov8n
│       └── run_<source>_<timestamp>/
│           ├── run_report.json  # Comprehensive run execution report
│           ├── images/          # Pristine frames for CVAT annotation
│           ├── previews/        # Highlighted frames for fast human review
│           └── metadata/        # JSON metadata for each candidate
├── reports/              # Global execution tracking (ignored by git)
│   ├── runs_history.json # JSON log of all historical runs
│   └── runs_history.csv  # CSV summary log for easy spreadsheet analysis
├── .gitignore            # Git ignore rules for caches, weights, venv, candidates, reports
└── README.md             # Project documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:duyanh030/traffic_counting.git
   cd traffic_counting
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install opencv-python ultralytics supervision streamlit
   ```

4. **Configure your camera credentials (if using RTSP):**
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your RTSP camera settings if needed.

---

## Usage Guide

### 1. Test RTSP Camera Connectivity
```bash
python test_rtsp.py
```

### 2. Run Vehicle Counting on Live RTSP Stream
```bash
python count_traffic.py
```

### 3. Run Vehicle Counting on an MP4 Video File
```bash
python count_traffic.py --source sample_video/EU.mp4
```

### 4. Headless Server Mode (Batch Processing)
Run without displaying the OpenCV GUI window (ideal for servers):
```bash
python count_traffic.py --source sample_video/EU.mp4 --no-view
```

### 5. Disable Candidate Capture
If you only need counting metrics without saving snapshot candidates:
```bash
python count_traffic.py --source sample_video/EU.mp4 --no-candidates
```

---

## Active Learning Review Studio (Streamlit UI)

To inspect and approve candidates captured during your runs:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### UI Capabilities:
1. **Session Selector (Sidebar):** Switch between models (`yolov8n`, etc.) and timestamped execution runs (`run_EU_20261119_...`).
2. **Metrics Dashboard:** Overview of total captured, pending, approved, and rejected snapshots.
3. **Filter Bar:** Filter candidates by:
   - Status (`pending`, `approved`, `rejected`)
   - Detection reason (`track_loss`, `low_confidence`)
   - Vehicle class (`car`, `motorcycle`, `bus`, `truck`)
4. **Batch Approval / Rejection:** Quickly approve or reject all visible filtered candidates with one click.
5. **Detail Inspector:** Compare pristine raw frames side-by-side with AI-highlighted preview frames and examine full bounding box JSON metadata.
6. **Execution History:** View global historical statistics from all previous runs (`reports/runs_history.json`).

---

## Configuration Reference

Settings can be customized directly in your `.env` file or in [config.py](config.py):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RTSP_USERNAME` | `admin` | Camera RTSP username |
| `RTSP_PASSWORD` | - | Camera RTSP password |
| `RTSP_IP` | `127.0.0.1` | Camera local IP address |
| `RTSP_PORT` | `554` | Camera RTSP port |
| `MODEL_NAME` | `yolov8n.pt` | YOLO model weights |
| `CONFIDENCE_THRESHOLD` | `0.35` | Object detection threshold |
| `IMAGE_SIZE` | `640` | Inference image resolution |
| `COUNTING_LINE_POSITION` | `0.6` | Height ratio for the virtual counting line (0.0 - 1.0) |
| `LOW_CONF_MIN` | `0.20` | Lower bound for uncertainty sampling |
| `LOW_CONF_MAX` | `0.38` | Upper bound for uncertainty sampling |
| `TRACK_LOSS_MIN_FRAMES`| `5` | Minimum tracked frames before flagging sudden disappearance |
| `SNAPSHOT_COOLDOWN_FRAMES` | `30` | Minimum frames between snapshots of the same object |
| `BORDER_MARGIN_PIXELS` | `35` | Frame edge margin to exclude normal vehicle exits |
