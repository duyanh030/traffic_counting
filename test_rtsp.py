import cv2
import os
import re

import config

# Configure RTSP transport over TCP to avoid frame tearing and packet loss on Wi-Fi
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def mask_url(url: str) -> str:
    """Mask password in RTSP URL for safe terminal logging."""
    return re.sub(r":([^@]+)@", ":****@", url)


print(f"Connecting to RTSP Camera: {mask_url(config.RTSP_URL)} ...")
cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Failed to open RTSP stream. Please check:")
    print(f" 1. Are the host machine and IP camera on the same network subnet ({config.RTSP_IP})?")
    print(" 2. Verify RTSP_USERNAME, RTSP_PASSWORD, RTSP_IP, and RTSP_PORT in your .env file.")
    print(" 3. Try appending /live or /stream to RTSP_PATH in your .env file.")
    exit()

print("✅ Successfully connected! Press 'q' on the video window to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed lost or interrupted...")
        break

    cv2.imshow("RTSP Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
