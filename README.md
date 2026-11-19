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
│  CVAT Server (cvat_integration.py)           │ ──► Auto Task Creation + AI Pre-Annotation
└──────────┬───────────────────────────────────┘
           │ (Human Labeling & Dataset Export)
           ▼
┌──────────────────────────────────────────────┐
│  retrain.py (Step 4: Continuous Training)    │ ──► Ingest -> Mosaic/HSV Aug -> Gate -> Deploy
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
  - **AI Pre-Annotation:** Automatically runs YOLO to pre-draw bounding boxes for obvious vehicles, saving ~85% annotation time.
- **Automated Retraining & Quality Gate (`retrain.py` & `dataset_manager.py`):**
  - Ingests CVAT YOLO 1.1 zip exports and manages 80/20 train/val dataset splits.
  - Advanced Data Augmentation: Mosaic (4-grid cross-stitching), MixUp, HSV lighting shifts, 3D perspective, and rotation.
  - **Two-Tier Quality Gate:** Evaluates validation $mAP@0.5 \ge 0.60$ and recall before model promotion.
  - **Zero-Downtime Hot Reload:** Automatically promotes validated weights to `models/yolov8_traffic_latest.pt` and increments version (`v1.0.0` -> `v1.1.0`).
  - **Automated Rollback:** Rejects inferior weights and preserves production stability if Quality Gate fails.

---

## Project Structure

```text
traffic_counting/
├── config.py             # Centralized project configuration module
├── candidate_detector.py # Active Learning: track-loss & uncertainty snapshot engine
├── count_traffic.py      # Main vehicle tracking, counting, and candidate capture
├── app.py                # Streamlit Web Studio for candidate review and CVAT push
├── cvat_integration.py   # CVAT REST API client & task upload pipeline
├── push_to_cvat.py       # Standalone CLI script to push approved candidates to CVAT
├── dataset_manager.py    # CVAT export zip ingestion & train/val split manager
├── retrain.py            # Step 4: Fine-tuning, Data Augmentation & Quality Gate
├── models/               # Model registry & version tracking
│   ├── version.json      # Semantic version history and validation metrics
│   └── yolov8_traffic_latest.pt # Active production model pointer
├── dataset/              # Training & validation splits (ignored by git)
│   ├── train/            # images/ and labels/
│   ├── val/              # images/ and labels/
│   └── traffic_data.yaml # YOLO dataset definition
├── test_rtsp.py          # Diagnostic utility to verify RTSP stream connectivity
├── .env.example          # Template for environment variables (committed to git)
├── .env                  # Local credentials and settings (ignored by git)
├── candidates/           # Auto-generated candidate snapshots per run (ignored by git)
├── reports/              # Global execution tracking & exports (ignored by git)
├── .gitignore            # Git ignore rules for caches, weights, venv, candidates, reports
└── README.md             # Project documentation
```

---

## Complete End-to-End Workflow

### Step 1: Run Traffic Counting & Capture Error Candidates
```bash
# Process an MP4 video (or leave --source empty for live RTSP)
python count_traffic.py --source sample_video/EU.mp4
```

### Step 2: Review & Triage in Streamlit Web Studio
```bash
streamlit run app.py
```
- Open `http://localhost:8501`.
- Inspect candidates in **Gallery View** or **Detail Inspector**.
- Click **✅ Approve** on genuine errors (occluded vehicles, ambiguous classes, missed detections).

### Step 3: Push Approved Candidates to CVAT
- On Streamlit, switch to the **"🚀 Push to CVAT"** tab.
- Click **[🚀 Push to CVAT Now]** (with Auto Pre-annotate enabled).
- Click the direct link to open the task in CVAT: all obvious vehicles are already pre-drawn!
- Fix the missing/mistaken vehicle and press **`Ctrl + S`**.
- Export dataset from CVAT: **Actions -> Export task dataset -> YOLO 1.1 (Save images checked)**.

### Step 4: Retrain, Quality Gate & Auto-Deploy
Run the automated retraining engine on the exported CVAT zip:
```bash
python retrain.py --zip /path/to/task_traffic_yolo_1.1.zip --epochs 30
```
- If Quality Gate passes ($mAP@0.5 \ge 0.60$):
  - Model is automatically promoted to `models/yolov8_traffic_latest.pt`.
  - Version updated in `models/version.json` (`v1.1.0`).
  - Next time you run `count_traffic.py`, the new improved model is loaded automatically!
- If Quality Gate fails:
  - System automatically executes rollback and keeps the existing stable model.
