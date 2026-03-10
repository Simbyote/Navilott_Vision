import cv2
import time

pipeline = (
    "libcamerasrc ! "
    "video/x-raw,colorimetry=bt709,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera pipeline")
    raise SystemExit(1)

count = 0
start = time.time()
last_report = start
duration = 60  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        count += 1
        now = time.time()

        if count % 30 == 0:
            elapsed = now - start
            fps = count / elapsed if elapsed > 0 else 0.0
            print(f"frames={count} fps={fps:.2f} shape={frame.shape}")

        if now - start >= duration:
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    cap.release()
