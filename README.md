# Real-time Traffic Counting System

An automated, real-time traffic detection, tracking, and counting application built with **YOLOv8**, **ByteTrack**, **Roboflow Supervision**, and **OpenCV**, processing live RTSP camera feeds.

---

## Features

- **Real-Time Vehicle Detection:** Powered by Ultralytics YOLOv8 (detects cars, motorcycles, buses, and trucks).
- **Multi-Object Tracking:** Native ByteTrack algorithm integration (`bytetrack.yaml`) for robust cross-frame ID persistence.
- **Directional Traffic Counting:** Uses Supervision's `LineZone` to count incoming (IN) and outgoing (OUT) vehicles across a virtual boundary.
- **Secure Configuration:** All sensitive camera credentials (IP, username, password, port) are kept in `.env` and never tracked by Git.
- **RTSP Over TCP Optimization:** Stable RTSP stream processing with low buffer latency over Wi-Fi/Ethernet.

---

## Project Structure

```text
traffic_counting/
├── config.py            # Centralized project configuration module
├── count_traffic.py     # Main vehicle tracking and counting application
├── test_rtsp.py         # Diagnostic utility to verify RTSP stream connectivity
├── .env.example         # Template for environment variables (committed to git)
├── .env                 # Local credentials and settings (ignored by git)
├── .gitignore           # Git ignore rules for caches, weights, and venv
└── README.md            # Project documentation
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

4. **Configure your camera credentials:**
   Copy `.env.example` to `.env` and enter your camera credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   RTSP_USERNAME=admin
   RTSP_PASSWORD=your_actual_password
   RTSP_IP=192.168.1.100
   RTSP_PORT=554
   RTSP_PATH=/
   ```

---

## Usage

### 1. Test RTSP Camera Connectivity
Verify your camera connection before starting the counting system:
```bash
python test_rtsp.py
```

### 2. Run Vehicle Counting
Launch the main traffic counting application:
```bash
python count_traffic.py
```
- Press **`q`** to stop the process and view the final traffic count summary in the terminal.

---

## Configuration Reference

Settings can be customized directly in your `.env` file or in [config.py](config.py):
- `RTSP_USERNAME`, `RTSP_PASSWORD`, `RTSP_IP`, `RTSP_PORT`, `RTSP_PATH`: RTSP connection credentials.
- `RTSP_URL`: Optional full URL override (if non-standard RTSP schema is used).
- `MODEL_NAME`: YOLO model weights (default: `yolov8n.pt`).
- `CONFIDENCE_THRESHOLD`: Object detection threshold (default: `0.35`).
- `IMAGE_SIZE`: Inference input resolution (default: `640`).
- `COUNTING_LINE_POSITION`: Vertical position of the counting line as a fraction of screen height (default: `0.6`).
