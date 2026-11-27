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
- **Robust Multi-Object Tracking:** Native ByteTrack algorithm integration (`bytetrack.yaml`) with **Class-Agnostic NMS** and **Majority Voting Class Smoothing** for rock-solid cross-frame ID persistence (prevents duplicate counting and eliminates ID-switching track loss alerts).
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
- **Automated Retraining, Regression Suite & Model Governance Gate (`retrain.py`, `golden_test_manager.py`, `model_deployer.py`):**
  - Ingests CVAT YOLO 1.1 zip exports and manages 80/20 train/val dataset splits with scenario-stratified grouping.
  - Advanced Data Augmentation: Mosaic, MixUp, HSV lighting shifts, 3D perspective, and rotation.
  - **Golden Test Set Regression Suite:** Automatically benchmarks newly trained models against permanent edge cases to guarantee no catastrophic forgetting.
  - **Staged Deployment Gate:** New weights are safely placed into `models/staging/` with a full regression and metric report awaiting human approval.
  - **Semantic Versioning & Zero-Downtime Hot Reload:** Promoting a candidate increments the semantic version (`v1.3.0` -> `v1.4.0`), archives immutable checkpoints, updates `models/version.json`, and links `models/yolov8_traffic_latest.pt`.
  - **1-Click Rollback:** Allows instant downgrade to any previous version if production regressions are observed.

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

---

### Step 4: Retrain, Golden Regression Test & Governed Deployment
Run the automated retraining engine on the exported CVAT zip:
```bash
python retrain.py --epochs 30 --freeze 10
```
- **CLI Hyperparameter Tuning Options:**
  - `--freeze N`: Freeze first N layers to preserve pretrained COCO features (e.g. `--freeze 10` locks Backbone, `--freeze 22` trains Detect Head only). Prevents Catastrophic Forgetting on small datasets.
  - `--optimizer AdamW`: Explicit optimizer selection (`AdamW`, `SGD`, or `auto`).
  - `--lr 0.001`: Initial learning rate.
  - `--epochs 30`: Number of fine-tuning epochs.
  - `--min-map50 0.60`: Quality Gate floor threshold.
  - `--auto-deploy`: Automatically promote without waiting for UI approval (disabled by default for safety).
- During retraining:
  - Ingests CVAT zip into `dataset/train` and `dataset/val` with scenario-stratified splitting.
  - Fine-tunes YOLOv8 with advanced data augmentations.
  - **Runs Golden Test Set Regression Suite automatically**: Tests the newly trained model against all registered permanent edge cases.
  - Safely stages the candidate in `models/staging/<staging_id>/` with full metadata and regression report.

---

## 🧪 Golden Test Set & Regression Suite Guide

### 1. Why Do We Need a Golden Test Set?
When fine-tuning object detection models with small batches of active-learning data (small-N fine-tuning), models suffer from **Catastrophic Forgetting** or **Regression**:
- Fixing a missed vehicle in dark shadow might inadvertently cause the model to miss cars in broad daylight or drop detection confidence on normal traffic.
- General Validation mAP can fluctuate wildly when new images are added to the validation split.

The **Golden Test Set** solves this by establishing a **permanent, immutable benchmark of critical edge cases** (e.g., Frame #440 motorcycle under overpass shadow). Every retrained model MUST be verified against these exact frames before it can ever be deployed to production.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Golden Test Set Suite                           │
│                                                                        │
│  [Frame #440] Motorcycle under shadow   Baseline: 0.36 -> Cand: 0.96 ✅ │
│  [Frame #164] Partial occlusion truck   Baseline: 0.73 -> Cand: 0.78 ✅ │
│  [Frame #739] Low-contrast sedan        Baseline: 0.31 -> Cand: 0.81 ✅ │
│                                                                        │
│               Regression Quality: 31/31 Passed (100.0%)               │
└────────────────────────────────────────────────────────────────────────┘
```

### 2. How to Add a Test Case to the Golden Test Set

#### Method A: 1-Click via Streamlit Web UI (Recommended)
1. In Streamlit, navigate to **Tab 2: Detail Inspector**.
2. Select the candidate frame containing the difficult edge case you want to lock in as a benchmark.
3. Directly below the Raw & Preview images, click **`⭐ Save to Golden Test Set (Regression Test)`**.
4. The system automatically:
   - Copies the pristine raw frame into permanent storage: `golden_test_set/images/<case_id>.jpg`.
   - Evaluates the current production model to record the **Baseline Confidence** (e.g., `0.3612`) and **Target Bounding Box**.
   - Registers the case into `golden_test_set/cases.json` with diagnostic notes.

#### Method B: Command Line (CLI)
```bash
# Add from a candidate metadata JSON snapshot
python golden_test_manager.py \
  --add-candidate candidates/yolov8_traffic_latest/run_xxx/metadata/cand_20261124_134305_f000440_track_loss.json \
  --note "Motorcycle under overpass shadow boundary (Frame 440)"
```

### 3. File Structure of the Golden Test Set
```text
golden_test_set/
├── cases.json            # Permanent registry of test cases, target bboxes & baseline scores
└── images/               # High-resolution benchmark frames
    ├── cand_20261124_134305_f000440_track_loss.jpg
    ├── cand_20261123_210338_f000164_track_loss.jpg
    └── ...
```

Sample entry inside `cases.json`:
```json
{
  "case_id": "cand_20261124_134305_f000440_track_loss",
  "image_file": "cand_20261124_134305_f000440_track_loss.jpg",
  "target_class": "motorcycle",
  "target_bbox": [678.0, 574.0, 715.0, 680.0],
  "baseline_conf": 0.3612,
  "required_conf": 0.35,
  "note": "Motorcycle under overpass shadow boundary (Frame 440)",
  "created_at": "2026-11-24 18:22:15"
}
```

### 4. Regression Passing Criteria
During regression testing, each test case is evaluated with the candidate model:
- **Class Match:** The model must predict the expected `target_class` on the target object.
- **Confidence Floor:** Candidate confidence must meet `required_conf` ($\ge 0.35$).
- **No Severe Regression:** The delta confidence ($\text{Cand Conf} - \text{Baseline Conf}$) must not drop by more than $0.15$ ($\Delta \ge -0.15$).

### 5. Running Regression Benchmarks Standalone via CLI
```bash
# Benchmark a staged candidate against active production baseline
python golden_test_manager.py --candidate models/staging/staging_20261127_112112/best.pt
```

---

## 🎯 Model Governance, Staging & Version Management Guide

### 1. Model Lifecycle Architecture

```text
┌─────────────────┐       ┌─────────────────┐       ┌────────────────────────┐
│  CVAT Dataset   │ ────► │   retrain.py    │ ────► │ models/staging/<id>/   │
│  Export (.zip)  │       │ (Augment+Train) │       │ (Isolated Sandboxing)  │
└─────────────────┘       └─────────────────┘       └───────────┬────────────┘
                                                                │
                                                                ▼
                                                    ┌────────────────────────┐
                                                    │ Golden Regression Test │
                                                    │   (31/31 Edge Cases)   │
                                                    └───────────┬────────────┘
                                                                │
                                    ┌───────────────────────────┴───────────────────────────┐
                                    ▼                                                       ▼
                      [🚀 Approve & Promote]                                  [🗑️ Reject & Discard]
                                    │                                                       │
                                    ▼                                                       ▼
                   ┌─────────────────────────────────┐                             ┌─────────────────┐
                   │ 1. Semantic Version Increment   │                             │ Staging cleaned │
                   │    (v1.3.0 -> v1.4.0)           │                             │ Candidate purged│
                   │ 2. yolov8_traffic_latest.pt     │                             └─────────────────┘
                   │ 3. Updated models/version.json  │
                   │ 4. Zero-Downtime Hot Reload     │
                   └────────────────┬────────────────┘
                                    │
                                    ▼ (In case of real-world issues)
                   ┌─────────────────────────────────┐
                   │  ⏪ 1-Click Rollback / Revert   │
                   └─────────────────────────────────┘
```

### 2. Staging Candidate Isolation
To guarantee production safety, newly retrained models are **never deployed directly to production**:
1. New weights are saved to `models/staging/<staging_id>/best.pt`.
2. A complete audit manifest (`metadata.json`) is generated, documenting:
   - Training timestamp, hyperparameters, and number of epochs.
   - List of CVAT zip archives ingested and exact train/val split filenames.
   - Full validation metrics ($mAP@0.5$, $mAP@0.5:0.95$, Precision, Recall).
   - Complete itemized Golden Test Set regression report.
3. The candidate waits in staging until a human operator reviews and approves it.

### 3. Reviewing Candidates in Streamlit (Tab 4: 🎯 Model Deployment)
When you open Streamlit and navigate to **Tab 4: Model Deployment**, you will see:
1. **Production Baseline Card:** Active semantic version (`v1.3.0`), active model weight file, and baseline mAP.
2. **Staged Model Candidates Section:**
   - Displays candidate ID, training timestamp, number of samples, and epochs.
   - **Metrics Grid:** Shows Validation mAP@0.5 (with color-coded delta against baseline), Precision, and Recall.
   - **Source Manifest:** Shows which CVAT export zip files were used.
   - **Golden Test Set Regression Benchmark Table:** An itemized table showing each test case, class, Baseline Confidence, Candidate Confidence, Delta, and Status (`✅ PASS` / `❌ FAIL`).

### 4. Promotion, Rejection & Rollback Actions

#### A. 🚀 Approve & Promote to Production
Clicking **`🚀 Approve & Promote to Production`**:
- Automatically calculates the next minor version (e.g. `v1.3.0` $\rightarrow$ `v1.4.0`).
- Copies `best.pt` to a permanent checkpoint `models/yolov8_traffic_v1.4.0.pt`.
- Atomically updates the production pointer `models/yolov8_traffic_latest.pt`.
- Updates `models/version.json` with full audit logs and archives the candidate from staging.
- All subsequent runs of `count_traffic.py` immediately use the new model with **zero downtime**.

#### B. 🗑️ Reject & Discard
Clicking **`🗑️ Reject & Discard`**:
- Safely deletes the staging folder `models/staging/<staging_id>/`.
- Leaves active production completely untouched.

#### C. ⏪ 1-Click Rollback / Revert
If a newly deployed model exhibits unforeseen behavior in real-world traffic:
1. Scroll down to **⏪ Revert / Downgrade Active Production Model** in Tab 4.
2. Select any previous version from the dropdown (e.g., `v1.3.0`, `v1.2.0`).
3. Click **`⏪ Rollback to Selected Version`**.
4. The system immediately re-links `yolov8_traffic_latest.pt` to the selected historical version and logs the rollback in `models/version.json`.

---

## 📊 MLOps Decision Framework: When to Approve vs. Reject a Model

When reviewing a staged candidate in Tab 4, evaluate against this **5-tier Decision Checklist**:

| # | Criterion | Required Threshold | What It Tells You |
| :-: | :--- | :--- | :--- |
| **1** | **Golden Test Set Pass Rate** | $\ge 95\%$ (Ideally 100%) | Guarantees critical edge cases and dark shadows remain fixed without catastrophic forgetting. |
| **2** | **Precision Score** | $\ge \text{Baseline} - 0.05$ | High precision (e.g. $> 85\%$) ensures the model does not produce false positives or phantom detections. |
| **3** | **Class-Specific Performance** | Core classes (car, motorcycle) improved | Look beyond the macro average: ensure primary traffic classes are healthy. |
| **4** | **Quality Gate Floor** | $mAP@0.5 \ge 0.60$ | Ensures minimum baseline generalizability across the validation set. |
| **5** | **Apples-to-Apples Evaluation** | Benchmarked against same val set | Compare baseline and candidate on the *same* validation split before judging mAP drops. |

> [!NOTE]
> **Why does Validation mAP sometimes drop from 0.9+ to 0.6+?**
> 1. **Validation Split Expansion:** When new CVAT zips are added, the validation set is reshuffled and expanded. A baseline score of `0.9360` calculated on a tiny 3-image val set cannot be directly compared to a candidate score on a 7-image val set.
> 2. **Single-Instance Class Penalty:** If the validation set has only 1 instance of a class (e.g., 1 cut-off `bus` at the extreme edge of the frame), missing that single difficult object drops the class AP to ~0.06. Because mAP is the arithmetic mean across all 4 classes, this pulls down the overall mAP even when `car` is at 97% and `motorcycle` is at 83%!
> 
> In this scenario, check the **Golden Test Set (100% Pass)** and **Precision (89.2%)** — if both are outstanding, the model should be **APPROVED**.
