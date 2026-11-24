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
- **Automated Root-Cause Diagnostic Assistant (`track_loss_diagnoser.py`):**
  - Instant geometric & confidence analysis to classify candidate errors:
    - **⭐ Partial Occlusion (Hard Positive):** Highlights vehicles partially obscured by large vehicles (Feature Occlusion) where humans can still see the vehicle $\rightarrow$ **Recommended: APPROVE**.
    - **Confidence Dipped / Severe Drop:** Uncovers false negatives where YOLO missed the vehicle $\rightarrow$ **Recommended: APPROVE**.
    - **Full Occlusion:** Detects vehicles completely hidden ($>80\%$) behind trucks/buses $\rightarrow$ **Recommended: REJECT** (avoids detector hallucination; handled by tracker buffer).
    - **Tracker ID Switch:** Distinguishes tracker discontinuities from detector failures $\rightarrow$ **Recommended: REJECT** (saves ~70% wasted annotation effort).
    - **Natural Exit:** Filters out expected tracking termination at camera frame boundaries $\rightarrow$ **Recommended: REJECT**.
- **Explainable AI (XAI) & Attention Heatmaps (`view_heatmap.py` & `view_channels.py`):**
  - **On-Demand Eigen-CAM Heatmaps:** Seamlessly integrated into Streamlit Tab 2 to visualize neural network feature activations (<50ms inference via cached model).
  - **Multi-Scale Layer Inspection:** Target specific network stages:
    - `Layer 15 (P3)`: Optimized for motorcycles, bicycles, and distant small objects.
    - `Layer 18 (P4)`: Optimized for medium-scale sedans and SUVs.
    - `Layer 21 (P5)`: Optimized for large trucks, buses, and containers.
    - `Layer 9 (SPPF)`: Explains macro environmental context and heavy shadow gradients.
  - **Diagnostic Identification:** Instantly differentiates **Feature Occlusion** (target features absorbed by larger neighbor) from **Texture Blending** (target blended into asphalt road background).
  - **Feature Channel Dissector:** Unpacks all sub-channel filters (e.g., 64 channels in Layer 15) into an interactive grid layout to view individual visual concepts (wheels, windshields, helmets).
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
├── config.py                 # Centralized project configuration module
├── candidate_detector.py     # Active Learning: track-loss & uncertainty snapshot engine
├── count_traffic.py          # Main vehicle tracking, counting, and candidate capture
├── app.py                    # Streamlit Web Studio for candidate review and CVAT push
├── track_loss_diagnoser.py   # Root-cause diagnostic engine (Partial vs Full Occlusion, ID Switch)
├── diagnose_candidate.py     # Standalone CLI tool to diagnose candidate JSON snapshots
├── view_heatmap.py           # Multi-layer Eigen-CAM attention heatmap generator
├── view_channels.py          # Sub-channel feature filter grid inspector (e.g. 64 channels)
├── cvat_integration.py       # CVAT REST API client & task upload pipeline
├── push_to_cvat.py           # Standalone CLI script to push approved candidates to CVAT
├── dataset_manager.py        # CVAT export zip ingestion & train/val split manager
├── golden_test_manager.py    # Golden Test Set registry & model regression test suite
├── retrain.py                # Step 4: Fine-tuning, Data Augmentation, Regression & Quality Gate
├── model_deployer.py         # Semantic versioning, staging gate & zero-downtime rollback
├── models/                   # Model registry & version tracking
│   ├── version.json          # Semantic version history and validation metrics
│   ├── staging/              # Staged model candidates awaiting human approval
│   └── yolov8_traffic_latest.pt # Active production model pointer
├── golden_test_set/          # Permanent regression test cases & benchmark images
│   ├── cases.json            # Regression test definitions and baseline confidences
│   └── images/               # Permanent benchmark images
├── dataset/                  # Training & validation splits (ignored by git)
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

### Step 2: Review, Diagnose, Save Golden Cases & Triage in Streamlit Web Studio
```bash
streamlit run app.py
```
- Open `http://localhost:8501`.
- Switch to **Detail Inspector** to view candidate frames:
  - **🩺 AI Smart Diagnostic Assistant:** Automatically detects the root cause of the alert (`Partial Occlusion`, `Detection Drop`, `Full Occlusion`, `ID Switch`) and displays a clear `🟢 APPROVE` or `⚪ REJECT` recommendation.
  - **⭐ Save to Golden Test Set (Regression Test):** Click this button on any tricky edge case (e.g. shadow transition at Frame #440) to save it permanently into `golden_test_set/`. The system records the baseline confidence (e.g. `0.32 < 0.35`). Any retrained model in the future MUST pass this benchmark!
  - **🔥 On-Demand Attention Heatmap:** Click **⚡ Generate Heatmap** on Layer 15 (Motorcycles), Layer 18 (Cars), or Layer 21 (Trucks) to inspect where the neural network placed its focus.
- Click **✅ Approve** on genuine errors (partially occluded vehicles, missed detections).
- Click **❌ Reject** on false alarms (tracker ID swaps, complete occlusions).

#### Optional: Terminal CLI Diagnostics, Heatmaps & Regression Suite
You can also run deep diagnostics, feature inspections, and regression evaluations directly from the command line:
```bash
# 1. Run root-cause diagnosis on any candidate JSON
python diagnose_candidate.py candidates/metadata/cand_20261119_100035_f005651_track_loss.json

# 2. Add a tricky candidate to Golden Test Set via CLI
python golden_test_manager.py --add-candidate path/to/metadata.json --note "Shadow transition"

# 3. Run regression benchmark comparing candidate model vs active baseline
python golden_test_manager.py --candidate models/staging/staging_xxxx/best.pt

# 4. Generate Eigen-CAM heatmaps across multiple YOLO layers simultaneously
python view_heatmap.py candidates/images/sample.jpg -l 9 15 18 21

# 5. Dissect and visualize all 64 sub-channel feature filters of Layer 15 in a grid
python view_channels.py candidates/images/sample.jpg -l 15 --cmap jet
```

### Step 3: Push Approved Candidates to CVAT
- On Streamlit, switch to the **"🚀 Push to CVAT"** tab.
- Click **[🚀 Push to CVAT Now]** (with Auto Pre-annotate enabled).
- Click the direct link to open the task in CVAT: all obvious vehicles are already pre-drawn!
- Fix the missing/mistaken vehicle and press **`Ctrl + S`**.
- Export dataset from CVAT: **Actions -> Export task dataset -> YOLO 1.1 (Save images checked)**.

### Step 4: Retrain, Golden Regression Test & Governed Deployment
Run the automated retraining engine on the exported CVAT zip:
```bash
python retrain.py --zip /path/to/task_traffic_yolo_1.1.zip --epochs 30
```
- During retraining:
  - Evaluates validation metrics on dataset split ($mAP@0.5$).
  - **Runs Golden Test Set Regression Suite automatically**: Tests the newly trained model against all registered edge cases (e.g. Frame #440) and reports the exact confidence increase (e.g. `0.32 -> 0.68`).
- Model Deployment & Governance Gate:
  - If Quality Gate passes, the model is staged in `models/staging/`.
  - Go to **Streamlit Tab "🎯 Model Deployment"**:
    - Inspect the **Golden Test Set Regression Benchmark Table** showing exact Old Conf vs New Conf and delta.
    - Click **🚀 Approve & Promote to Production** to deploy with zero downtime.
    - If needed, instantly **⏪ Rollback** to any historical version with 1 click.
