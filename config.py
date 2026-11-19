import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent


def load_env(env_path: Path):
    """Load environment variables from a .env file without external dependencies."""
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


# Load .env file automatically if present
load_env(BASE_DIR / ".env")

# Camera connection credentials
RTSP_USERNAME = os.getenv("RTSP_USERNAME", "")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", "")
RTSP_IP = os.getenv("RTSP_IP", "127.0.0.1")
RTSP_PORT = os.getenv("RTSP_PORT", "554")
RTSP_PATH = os.getenv("RTSP_PATH", "/")


def build_rtsp_url() -> str:
    """Build or retrieve the RTSP connection URL."""
    custom_url = os.getenv("RTSP_URL")
    if custom_url:
        return custom_url

    path = RTSP_PATH if RTSP_PATH.startswith("/") else f"/{RTSP_PATH}"
    if RTSP_USERNAME and RTSP_PASSWORD:
        return f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}{path}"
    return f"rtsp://{RTSP_IP}:{RTSP_PORT}{path}"


RTSP_URL = build_rtsp_url()

# COCO vehicle class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Detection & Tracking parameters
LATEST_MODEL_PATH = BASE_DIR / "models" / "yolov8_traffic_latest.pt"
DEFAULT_MODEL = str(LATEST_MODEL_PATH) if LATEST_MODEL_PATH.is_file() else "yolov8n.pt"
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
COUNTING_LINE_POSITION = float(os.getenv("COUNTING_LINE_POSITION", "0.6"))

# Candidate Detector / Active Learning parameters
CANDIDATES_DIR = BASE_DIR / "candidates"
REPORTS_DIR = BASE_DIR / "reports"
LOW_CONF_MIN = float(os.getenv("LOW_CONF_MIN", "0.20"))
LOW_CONF_MAX = float(os.getenv("LOW_CONF_MAX", "0.38"))
TRACK_LOSS_MIN_FRAMES = int(os.getenv("TRACK_LOSS_MIN_FRAMES", "5"))
SNAPSHOT_COOLDOWN_FRAMES = int(os.getenv("SNAPSHOT_COOLDOWN_FRAMES", "30"))
BORDER_MARGIN_PIXELS = int(os.getenv("BORDER_MARGIN_PIXELS", "35"))

# CVAT Annotation Server Configuration
CVAT_URL = os.getenv("CVAT_URL", "http://localhost:8080").rstrip("/")
CVAT_USERNAME = os.getenv("CVAT_USERNAME", "")
CVAT_PASSWORD = os.getenv("CVAT_PASSWORD", "")
