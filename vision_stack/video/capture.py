import cv2
import time

pipeline = (
    "libcamerasrc ! "
    "video/x-raw,colorimetry=bt709,width=480,height=360,framerate=20/1 ! "
    "videoconvert ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera pipeline")
    raise SystemExit(1)

# Writer initialization 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (480, 360))

count = 0
start = time.time()
duration = 60  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        out.write(frame)          # save each frame
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
    out.release()                 # Flush and close the file
    print("Saved to output.avi")