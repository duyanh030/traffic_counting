# Real-time Traffic Counting & Active Learning System

An automated, end-to-end traffic detection, tracking, counting, and active learning candidate capture system built with **YOLOv8**, **ByteTrack**, **Roboflow Supervision**, **CVAT REST API**, **Streamlit**, and **OpenCV**, supporting both live **RTSP camera feeds** and **MP4 video files**.

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
           │ (Push Approved Candidates)
           ▼
┌──────────────────────────────────────────────┐
│  CVAT Server (cvat_integration.py)           │ ──► Auto Task Creation & Image Upload
└──────────┬───────────────────────────────────┘
           │ (Annotation & Dataset Export)
           ▼
┌──────────────────────────────────────────────┐
│  Retraining Pipeline (retrain.py - Step 4)   │ ──► Fine-tune -> Benchmark Gate -> Deploy
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
- **1-Click CVAT Integration (`cvat_integration.py` & `push_to_cvat.py`):**
  - Directly authenticates with self-hosted CVAT (`http://localhost:8080`) via REST API v2.
  - Automatically creates tasks with vehicle label categories (`car`, `motorcycle`, `bus`, `truck`).
  - Seamlessly uploads approved candidate images packaged in a zip archive.
  - Provides offline local ZIP export fallback for manual import.
- **Partitioned Execution Sessions & Reports:**
  - Candidates are organized into distinct folders by **model version** and **timestamped run session** (`candidates/<model>/run_<source>_<timestamp>/`).
  - Automated generation of session-specific `run_report.json` and global history logs (`reports/runs_history.json` and `reports/runs_history.csv`).
- **Secure Configuration:** Sensitive camera and CVAT credentials (URL, username, password) are kept in `.env` and never tracked by Git.

---

## Project Structure

```text
traffic_counting/
├── config.py             # Centralized project configuration module
├── candidate_detector.py # Active Learning: track-loss, uncertainty snapshot & reporting engine
├── count_traffic.py      # Main vehicle tracking, counting, and candidate capture
├── app.py                # Streamlit Web Studio for candidate review and CVAT push
├── cvat_integration.py   # CVAT REST API client & task upload pipeline
├── push_to_cvat.py       # Standalone CLI script to push approved candidates to CVAT
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
├── reports/              # Global execution tracking & exports (ignored by git)
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
   pip install opencv-python ultralytics supervision streamlit requests
   ```

4. **Configure your camera and CVAT credentials:**
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your RTSP camera and CVAT server settings:
   ```env
   # RTSP Camera
   RTSP_USERNAME=admin
   RTSP_PASSWORD=your_camera_password
   RTSP_IP=192.168.1.100
   RTSP_PORT=554
   RTSP_PATH=/

   # CVAT Server
   CVAT_URL=http://localhost:8080
   CVAT_USERNAME=admin
   CVAT_PASSWORD=your_cvat_password
   ```

---

## Usage Guide

### 1. Run Vehicle Counting on Video or RTSP Stream
```bash
# On an MP4 video file
python count_traffic.py --source sample_video/EU.mp4

# On a live RTSP stream
python count_traffic.py

# Headless server mode
python count_traffic.py --source sample_video/EU.mp4 --no-view
```

### 2. Launch Review Studio (Streamlit Web UI)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
- Review candidate snapshots in **Gallery View** or **Detail Inspector**.
- Click **✅ Approve** for valuable error cases (occlusions, misclassifications, ambiguous vehicles).

### 3. Push Approved Candidates to CVAT (Step 3)

**Option A: 1-Click via Streamlit Web Studio:**
1. Open the **"🚀 Push to CVAT"** tab in the Streamlit UI.
2. Confirm the CVAT credentials and task name.
3. Click **[🚀 Push to CVAT Now]**.
4. Click the direct link generated on screen to open the task in CVAT and start labeling!

**Option B: Via Command Line (CLI):**
```bash
# Push approved candidates from default candidates folder
python push_to_cvat.py

# Push approved candidates from a specific run session
python push_to_cvat.py --run-dir candidates/yolov8n/run_EU_20261119_100948

# Export to a local ZIP archive instead (Offline mode)
python push_to_cvat.py --export-zip-only reports/approved_dataset.zip
```
