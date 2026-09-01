# Pipeline Specification

> Phases 1–3: Camera Acquisition → Vision Perception → Navigation Signal Processing

---

## Table of Contents

- [Pipeline Phase 1: Camera Acquisition](#pipeline-phase-1-camera-acquisition)
- [Pipeline Phase 2: Vision Perception](#pipeline-phase-2-vision-perception)
- [Pipeline Phase 3: Navigation Signal Processing](#pipeline-phase-3-navigation-signal-processing)

---

## Pipeline Phase 1: Camera Acquisition

> *Photons → Frame Buffers*

### Camera Hardware Path

```
Environment
    │
    ▼ (photons)
IMX219 CMOS Sensor          ← lens focuses light onto sensor grid
    │
    ▼ (RAW Bayer image)
CSI-2 Interface             ← high-speed, low-latency transfer to processor
    │
    ▼
ISP / DMA Transfer
    │
    ▼
Raspberry Pi Camera Driver  ← kernel-level driver; manages sensor timing,
    │                          exposure, and frame organization
    ▼
libcamera Framework         ← manages camera pipeline, memory buffers,
    │                          and image processing configuration
    ▼
GStreamer Pipeline Bridge   ← libcamerasrc → videoconvert → appsink
    │                          bridges libcamera into the media pipeline
    ▼
OpenCV (cv2.VideoCapture)   ← receives GStreamer frames, converts to
    │                          NumPy/OpenCV image matrices
    ▼
Python Application Layer    ← executes the perception pipeline
```

### Camera Software Stack (Layers)

| Layer | Component                  | Role                                                              |
|-------|----------------------------|-------------------------------------------------------------------|
| 1     | IMX219 Sensor              | Captures raw Bayer image data via CSI from Pi Camera Module v2    |
| 2     | Raspberry Pi Camera Driver | Kernel-level driver: interfaces directly with CSI camera hardware |
| 3     | libcamera Framework        | Manages camera pipeline, buffers, and image processing config     |
| 4     | GStreamer Pipeline Bridge  | Bridges libcamera into a 480×360 @ ~20 FPS media pipeline         |
| 5     | OpenCV Interface           | Receives GStreamer appsink frames → NumPy/OpenCV matrices         |
| 6     | Python Perception App      | Executes all pipeline stages                                      |

### Capture Configuration

```python
# GStreamer pipeline string (cv2.VideoCapture)
pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=480, height=360, framerate=20/1 ! "
    "videoconvert ! "
    "appsink drop=true max-buffers=1 sync=false"
)
```

| Parameter    | Value                                    |
|--------------|------------------------------------------|
| Resolution   | 480×360 (operating point)                |
| Frame rate   | ~17–20 FPS (accepted operating point)    |
| Frame format | YUV (libcamera native); BGR on request   |
| Buffering    | Ring buffer, 1 frame max                 |

> **Operating point note:** 480×360 @ ~17–20 FPS was accepted by Dr. Stevens as the target after profiling on hardware. The 50 ms per-frame budget (~20 FPS) provides ~26 ms headroom after pipeline stages complete. Earlier documentation targeting 640×480 @ 30 FPS is preserved in the archive.

**Output:** Continuous stream of timestamped YUV image frames passed to Phase 2.

---

## Pipeline Phase 2: Vision Perception

> *Converts pixels into detections*

### Phase 2 Pipeline

```
YUV Frame (from Phase 1)
        │
        ▼
┌───────────────────────┐
│   Preprocessing       │  ← histogram equalization (undistortion optional — see note)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   ROI Cropping        │  ← reduces computation and false detection rate
└──────┬────────────────┘
       │
       ├──────────────────────────────┐
       ▼                              ▼
┌─────────────────┐        ┌──────────────────────┐
│  Color Branch   │        │  Geometry Branch     │
│  (HSV)          │        │  (Grayscale)         │
│                 │        │                      │
│ HSV Threshold   │        │ Canny Edge Detection │
│ Binary Masks    │        │ Contour Extraction   │
│ Traffic Light   │        │ Lane Bounds /        │
│ State Detection │        │ Stop Sign Detection  │
└────────┬────────┘        └──────────┬───────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
        ┌───────────────────────┐
        │  Feature Fusion &     │
        │  Labelling            │  ← merge detections, apply bounding
        └───────────┬───────────┘     boxes, confidence scores, timestamps
                    │
                    ▼
        ┌───────────────────────┐
        │  Lane Offset          │
        │  Estimation           │  ← pixel-based lateral offset
        └───────────┬───────────┘     normalized output [-1.0, +1.0]
                    │
                    ▼
             Phase2Output
```

### Preprocessing

- **Histogram Equalization** — performed in YCrCb space; stabilizes intensity distribution under varying lighting. This is the primary conditioning step and is always active.
- **Lens Undistortion** *(optional)* — corrects barrel distortion using the camera matrix from Step 1 calibration. The IMX219 exhibits modest distortion, and because the pipeline uses pixel-based (non-metric) lateral offset the practical benefit in the center ROI is limited. A comparative test on course frames is recommended before enabling in production. If enabled, it adds compute overhead per frame. Calibration procedure is in `operations.md` Step 1.

### Color Branch (HSV Thresholding)

HSV color space separates color from brightness, making detection more robust under changing lighting conditions. Binary masks identify specific traffic light states (red / yellow / green). Blob-size filtering rejects false positives from reflections and background colors.

### Geometry Branch (Grayscale + Edge Detection)

ROI frames undergo Canny edge detection and contour extraction to find lane boundaries and stop sign octagons. Contours are filtered by area bounds and elongation ratio — lane markings are long and thin relative to background clutter. `cv2.minAreaRect` orientation-independent elongation is used so both horizontal and vertical lane lines pass the filter.

### Feature Fusion & Spatial Awareness

Both branches merge to produce normalized detection objects: bounding boxes, confidence scores, and timestamps. Conflict resolution keeps the single highest-confidence traffic light candidate and stop sign candidate per frame; all lane boundary candidates are kept.

### Lane Offset Estimation (Pixel-Based)

```
lane_center = (left_x + right_x) / 2.0
offset      = (lane_center − frame_center) / (lane_width_px / 2.0)
offset      = clamp(offset, −1.0, +1.0)
```

Lateral offset is computed directly from boundary pixel x-positions in the lane ROI. No homography or metric transform is required. This is appropriate for the fixed, forward-facing, low-mount camera geometry where a consistent pixel-to-steering mapping can be assumed.

> **Why not homography:** The camera mount is fixed at 3–4 cm ground level, forward-facing. A bird's-eye transform requires accurate physical calibration of mount height and tilt, adds ~3–5 ms per frame, and produces metric outputs that need re-normalizing for the steering PID anyway. Pixel-based offset is simpler and more robust for this geometry. The homography calibration procedure is preserved in `operations.md` Step 3 for reference.

### Phase 2 Output Format

```
Phase2Output
    detections       list[DetectionObject]
    lane_offset      LaneOffsetResult
    frame_id         int
    timestamp_ms     int
    detection_count  int
```

`lane_offset` is always populated. If no boundary candidates were detected, `mode` is `"none"` and `offset` is `0.0`. Phase 3 must check `mode` before using `offset` as a steering signal.

### Coordinate Frame Definition

Position values in detection objects use the following coordinate system:

```
        +Y (forward)
         │
         │
         │
─────────┼─────────  origin: camera center (image plane)
         │
        -Y
    -X ──┼── +X (right)
```

| Field  | Definition                                              |
|--------|---------------------------------------------------------|
| Origin | Camera center projected onto the lane ROI               |
| X axis | Horizontal image plane — positive right                 |
| Y axis | Forward direction from robot — positive away from robot |
| Units  | Pixels (lane offset normalized to [-1.0, +1.0])         |
| Z axis | Not used — flat ground plane assumed                    |

**Why this matters:** `lane_offset` drives the Phase 3 PID steering input. A positive offset means the robot is left of lane center and must steer right. A sign error here propagates directly into incorrect steering commands.

---

## Pipeline Phase 3: Navigation Signal Processing

> *Converts detections to decisions*

### Phase 3 Pipeline

```
Detection Objects (from Phase 2)
          │
          ▼
┌─────────────────────────┐
│  Motion Tracking &      │  ← object tracked via motion vector for
│  Consistency Check      │    stability in lanes and signs
└──────────┬──────────────┘    movement proportional to robot speed
           │
           ▼
┌─────────────────────────┐
│  Temporal Filtering     │  ← detections smoothed across frames
│  (frames N, N+1, N+2)   │    (e.g. averaging) to reduce noise
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Confidence             │  ← reject weak / unreliable detections
│  Thresholding Filter    │    prevents noise-driven false positives
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  State Estimation                                           │
│  (Actionable Commands)                                      │
│                                                             │
│  Vision estimates (from above)    ──────────────────────┐   │
│  Encoder odometry (speed, dist)   ──────────────────┐   │   │
│  Servo rotational feedback        ──────────────┐   ├───┴───┤
│  IMU (yaw rate, accel)            ──────────┐   ├───┘       │
│                                             └───┘   Fused   │
│                                                    Estimate │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
   Validated Navigation Signals
```

### Sensor Inputs to State Estimation

Three hardware sensor streams feed directly into State Estimation alongside the vision-derived detections. They do not replace any existing stage — they augment the final estimation step only.

#### Encoder — Odometry Metrics

The encoder is sampled each frame cycle and provides distance-domain context to corroborate vision-derived position estimates.

| Metric              | Description                                    | Units | Role in Estimation                                                             |
|---------------------|------------------------------------------------|-------|--------------------------------------------------------------------------------|
| `wheel_speed`       | Instantaneous linear velocity                  | m/s   | Scales motion tracking vectors; sanity-checks lane offset rate of change       |
| `distance_traveled` | Cumulative odometric distance since last reset | m     | Dead-reckoning fallback when vision detections fall below confidence threshold |

**Sampling note (Pi-only):** The encoder is read via GPIO interrupt or polling in the same Python process. Attach the counter to a high-priority thread (`os.nice(-5)`) to minimize missed pulses under perception load. Timestamps must align with the frame `timestamp_ms` used throughout the pipeline.

#### Servo — Rotational Feedback

The servo reports its actual achieved shaft angle, closing the loop between commanded heading and physical steering response.

| Metric                  | Description                                 | Units   | Role in Estimation                                                        |
|-------------------------|---------------------------------------------|---------|---------------------------------------------------------------------------|
| `servo_angle_actual`    | Measured shaft position from servo feedback | degrees | Compared against `heading_error` command to detect mechanical lag or slip |
| `servo_angle_commanded` | Last angle written to the servo             | degrees | Retained in state for delta computation each cycle                        |

The difference `servo_angle_commanded − servo_angle_actual` is the **actuation error** (`servo_delta`). If this delta exceeds a defined threshold across consecutive frames, State Estimation flags a degraded steering condition — reflected in the status field of the navigation signal packet.

#### IMU — Inertial Sensing

The IMU feeds directly into State Estimation as a high-rate, vision-independent source of heading rate and lateral disturbance.

| Metric          | Description                                  | Units | Role in Estimation                                                               |
|-----------------|----------------------------------------------|-------|----------------------------------------------------------------------------------|
| `yaw_rate`      | Angular velocity about the vertical axis     | deg/s | Integrates between frames to propagate heading estimate when vision is degraded  |
| `lateral_accel` | Linear acceleration along the robot's X axis | m/s²  | Cross-checks unexpected lateral movement (e.g. wheel slip, surface irregularity) |

**Integration note:** The IMU operates at a rate higher than the 30 FPS camera loop (typically 100–200 Hz on common I²C parts). Within each frame cycle, the IMU driver accumulates samples and presents the **mean `yaw_rate`** and **peak `lateral_accel`** for that frame window to State Estimation. This avoids per-sample processing burden on the main loop.

**Pi-only timing caveat:** I²C polling on the Pi is not interrupt-driven in Python. Budget ~0.5–1 ms for an I²C burst read per frame. This is within the existing frame budget headroom documented in `phase2_pipeline.md`.

---

### State Estimation Outputs

The sensor inputs above augment the final state estimation step.

| Feature         | Description                              | Primary Source         | Augmented By                                       |
|-----------------|------------------------------------------|------------------------|----------------------------------------------------|
| `lane_offset`   | Lateral distance from lane center        | Vision (pixel-based)   | EMA + IMU dead-reckoning when confidence < threshold |
| `heading_error` | Angular deviation from target heading    | Vision (lane geometry) | IMU yaw rate integration; servo actuation error    |
| `traffic_state` | Current signal: stop / proceed / caution | Vision (color branch)  | — (vision-only; no inertial equivalent)            |

### Validated Navigation Signal Packet

```
┌───────────────────────────────────────┐
│  Validated Navigation Signals         │
│  ┌──────────────────┬───────────────┐ │
│  │ Validated        │ Status        │ │
│  │ Coordinates      │               │ │
│  ├──────────────────┴───────────────┤ │
│  │ Inertial State                   │ │
│  │   yaw_rate       (deg/s)         │ │
│  │   lateral_accel  (m/s²)          │ │
│  │   wheel_speed    (m/s)           │ │
│  │   servo_delta    (degrees)       │ │
│  ├──────────────────────────────────┤ │
│  │ Global Timestamp                 │ │
│  └──────────────────────────────────┘ │
└───────────────────────────────────────┘
```

Final perception outputs are packed into a structured format for navigation:

- validated coordinates
- object status
- inertial state (`yaw_rate`, `lateral_accel`, `wheel_speed`, `servo_delta`)
- global timestamp

`servo_delta` = `servo_angle_commanded − servo_angle_actual`, carried as a diagnostic field each cycle.
