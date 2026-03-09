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
| 4     | GStreamer Pipeline Bridge  | Bridges libcamera into a 640×480 @ 30 FPS media pipeline          |
| 5     | OpenCV Interface           | Receives GStreamer appsink frames → NumPy/OpenCV matrices         |
| 6     | Python Perception App      | Executes all pipeline stages                                      |

### Capture Configuration

```python
# GStreamer pipeline string (cv2.VideoCapture)
pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! "
    "appsink drop=true max-buffers=1 sync=false"
)
```

| Parameter   | Value                       |
|-------------|-----------------------------|
| Resolution  | 640×480 (standard)          |
| Frame rate  | 10–30 FPS (target: 30 FPS)  |
| Frame format| BGR (OpenCV native)         |
| Buffering   | Ring buffer, 1 frame max    |

**Output:** Continuous stream of timestamped BGR image frames passed to Phase 2.

---

## Pipeline Phase 2: Vision Perception

> *Converts pixels into detections*

### Phase 2 Pipeline

```
BGR Frame (from Phase 1)
        │
        ▼
┌───────────────────────┐
│   Preprocessing       │  ← undistortion, histogram equalization
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
        │  Perspective          │
        │  Transformation       │  ← homography → bird's-eye view
        └───────────┬───────────┘     for lateral offset computation
                    │
                    ▼
             Detection Object
```

### Preprocessing

Performed on both top and bottom ROI regions:

- **Undistortion** — camera calibration model corrects lens distortion; without this, lane lines curve incorrectly and
perspective transforms become inaccurate.
- **Histogram Equalization** — stabilizes pixel intensity distribution; improves edge detection and HSV color segmentation
visibility under varying lighting.

### Color Branch (HSV Thresholding)

HSV color space separates color from brightness, making detection more robust under changing lighting conditions. Binary masks
identify specific traffic light states (red / yellow / green). Blob-size filtering rejects false positives from brake lights,
reflections, and road signs.

### Geometry Branch (Grayscale + Edge Detection)

ROI frames undergo Canny edge detection and contour extraction to find lane boundaries and stop sign octagons. Edges represent
intensity discontinuities corresponding to physical boundaries in the environment.

### Feature Fusion & Spatial Awareness

Both branches merge to extract object attributes: bounding boxes, confidence scores, and timestamps on each detected candidate
(traffic lights, stop signs, lane lines, other objects).

### Perspective Transformation (Homography)

```
Camera View (perspective)    →    Bird's-Eye View (top-down)
    ┌──────────────┐                 ┌──────────────┐
    │   /──────\   │                 │ ──────────── │
    │  /  lane  \  │    homography   │              │
    │ /          \ │   ──────────▶   │ ──────────── │
    └──────────────┘                 └──────────────┘
```

Converts pixel coordinates into a real-world coordinate system for lateral offset calculation. Z-plane is assumed constant. Works best in planar scenarios with a fixed camera mount.

### Phase 2 Output Format

```json
{
  "type":       "traffic_light | stop_sign | lane_boundary",
  "position":   { "x": float, "y": float },
  "confidence": float,
  "timestamp":  int
}
```

**Note:** May set `float` values to `integer` to reduce RAM usage; cheaper to create integer arrays than float arrays.

### Coordinate Frame Definition

Position values in the detection object use the following coordinate system:

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

| Field     | Definition                                              |
|-----------|---------------------------------------------------------|
| Origin    | Camera center projected onto the ground plane           |
| X axis    | Horizontal image plane — positive right                 |
| Y axis    | Forward direction from robot — positive away from robot |
| Units     | Meters (after homography transform)                     |
| Z axis    | Assumed constant (flat ground plane)                    |

**Why this matters:** Lane offset and heading error in Phase 3 are computed from these coordinates. A sign error or unit mismatch here propagates directly into
incorrect steering commands.

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
┌─────────────────────────┐
│  State Estimation       │  ← compute final navigation features
│  (Actionable Commands)  │    converts pixel data → nav signals
└──────────┬──────────────┘
           │
           ▼
   Validated Navigation Signals
```

### State Estimation Outputs

| Feature             | Description                                |
|---------------------|--------------------------------------------|
| Lane offset         | Lateral distance from lane center          |
| Heading error       | Angular deviation from target heading      |
| Traffic light state | Current signal: stop / proceed / caution   |

### Validated Navigation Signal Packet

```
┌───────────────────────────────────────┐
│  Validated Navigation Signals         │
│  ┌──────────────────┬───────────────┐ │
│  │ Validated        │ Status        │ │
│  │ Coordinates      │               │ │
│  ├──────────────────┴───────────────┤ │
│  │ Global Timestamp                 │ │
│  └──────────────────────────────────┘ │
└───────────────────────────────────────┘
```

Final perception outputs are packed into a structured format for navigation:

- validated coordinates
- object status
- global timestamp.
