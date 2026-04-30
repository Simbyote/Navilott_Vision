# Vision Processing Pipeline — Phase 2

This document describes the Phase 2 perception pipeline used in the Navilott vision system. Phase 2 converts a raw camera frame into structured detections and geometric information for use by the navigation system in Phase 3.

The pipeline is implemented as a series of deterministic processing stages with explicit input/output contracts.

---

## Pipeline Overview

```
Raw Frame
   │
   ▼
Preprocess
   │
   ▼
ROI Crop
   │
   ├──────── Color Branch ──── TrafficLightCandidates ──┐
   │                                                     │
   └──────── Geometry Branch ── LaneCandidates ──────────┤
                                SignCandidates ──────────┤
                                                         ▼
                                                 Feature Fusion
                                                         │
                                                  DetectionObjects
                                                         │
                                                         ▼
                                               Lane Offset Estimation
                                                         │
                                                         ▼
                                                  Phase2Output
                                                         │
                                                         ▼
                                                     Phase 3
                                                  (Navigation)
```

---

## Stage 1 — Preprocessing

**File:** `preprocess.py`

Preprocessing conditions the raw camera frame before any spatial analysis occurs.

### Operations

#### Histogram Equalization

Equalization is performed in YCrCb space so that intensity contrast improves while color information remains intact.

```
YUV → BGR
BGR → YCrCb
equalize(Y channel)
YCrCb → BGR
BGR → YUV
```

#### Gaussian Blur

Suppresses sensor noise before edge detection and HSV thresholding.

```
GaussianBlur(kernel=(5,5))
```

### Output

```
np.ndarray (H, W, 3)
dtype: uint8
color: YUV
```

---

## Stage 2 — ROI Cropping

**File:** `roi_crop.py`

The frame is partitioned into three regions of interest (ROIs) before branching.

### ROI Layout

| ROI           | Frame Region        | Purpose                 |
|---------------|---------------------|-------------------------|
| `lane_roi`    | lower half of frame | lane boundary detection |
| `traffic_roi` | top-center region   | traffic light detection |
| `sign_roi`    | right half of frame | stop sign detection     |

### Visualization

```
┌─────────────┬─────────────┬─────────────┐
│             │ traffic_roi │             │
│             │             │             │
├─────────────┴─────────────┴─────────────┤
│                                         │
│                lane_roi                 │
│                                         │
└─────────────────────────────────────────┘
```

> `sign_roi` occupies the right half of the full frame height and overlaps both horizontal bands. It is not shown in the 2D layout above.

### Output

```
ROICropResult
    lane_roi
    traffic_roi
    sign_roi
    lane_rect
    traffic_rect
    sign_rect
    frame_id
    source_shape
```

Each ROI is a view of the original frame (no memory copy).

---

## Stage 3A — Color Branch

**File:** `color_branch.py`

Detects traffic light states using HSV color segmentation.

### Pipeline

```
YUV ROI
  ↓
HSV conversion
  ↓
HSV threshold masks
  ↓
Connected component extraction
  ↓
Blob filtering
```

### HSV Masks

Three color masks are generated: `red`, `yellow`, `green`.

Red uses two ranges because hue wraps in HSV space:

- `0°–10°`
- `170°–180°`

These masks are combined:

```
mask_red = mask_low OR mask_high
```

### Blob Filtering

Each detected blob is filtered using area threshold and aspect ratio bounds.

### Output

```
TrafficLightCandidate
    label        ("red" | "yellow" | "green")
    bbox         (x, y, w, h)
    confidence
    frame_id
    timestamp_ms
```

Confidence is based on normalized blob area.

---

## Stage 3B — Geometry Branch

**File:** `geometry_branch.py`

Detects lane boundaries and stop sign shapes using contour geometry.

### Pipeline

```
YUV ROI
  ↓
Grayscale conversion
  ↓
Canny edge detection
  ↓
Contour extraction
  ↓
Geometric filtering
```

### Lane Boundary Detection

Contours are filtered based on area limits and elongation ratio. Lane lines must be long relative to their width.

#### Output

```
LaneCandidate
    label
    bbox
    contour
    confidence
    frame_id
    timestamp_ms
```

### Stop Sign Detection

Stop signs are detected using polygon approximation:

```
cv2.approxPolyDP(contour)
```

Expected result: ≈ 8 vertices. Confidence combines vertex proximity to 8 and contour area.

#### Output

```
SignCandidate
    label
    bbox
    contour
    vertex_count
    confidence
    frame_id
    timestamp_ms
```

---

## Stage 4 — Feature Fusion

**File:** `feature_fusion.py`

Combines results from the color branch and geometry branch into a unified detection format.

### Responsibilities

- Normalize detection formats
- Resolve conflicts within each detection class
- Compute bounding box centroids

### Detection Schema

```json
{
  "type":       "traffic_light | stop_sign | lane_boundary",
  "position":   { "x": float, "y": float },
  "confidence": float,
  "timestamp":  int
}
```

Position is the centroid of the bounding box.

### Conflict Resolution Rules

| Class           | Rule                                                                      |
|-----------------|---------------------------------------------------------------------------|
| `traffic_light` | Only one color may exist per frame. Keep the highest confidence candidate |
| `lane_boundary` | All candidates are kept. Multiple lane boundaries are valid               |
| `stop_sign`     | Only the strongest candidate is kept. Keep the highest confidence         |

---

## Stage 5 — Lane Offset Estimation

**File:** `lane_offset.py`

Computes the robot's normalized lateral offset from lane center using the
pixel x-positions of lane boundary candidates produced by feature fusion

### Computation

Lane boundary candidates are filtered to those meeting the confidence
threshold. The highest-confidence left and right anchors are selected by
x-position. Lane center is their midpoint. Offset is normalized to the lane
half-width:

```
    lane_center = (left_x + right_x) / 2.0
    offset      = (lane_center − frame_center) / (lane_width_px / 2.0)
    offset      = clamp(offset, −1.0, +1.0)
```

A positive offset means the robot is left of lane center. A negative offset
means the robot is right of lane center

### Detection Modes

| Mode             | Condition                                      | Offset source                        |
|------------------|------------------------------------------------|--------------------------------------|
| `two_boundary`   | Left and right anchors detected                | Midpoint formula above               |
| `left_only`      | Only left boundary detected                    | Inferred from left anchor position   |
| `right_only`     | Only right boundary detected                   | Inferred from right anchor position  |
| `width_rejected` | Boundary span below `min_lane_width_px`        | Midpoint used; anchors discarded     |
| `none`           | No candidates above confidence threshold       | offset = 0.0; Phase 3 uses fallback  |

Phase 3 reads `mode` to determine whether to trust `offset` directly or
substitute a dead-reckoned estimate from EMA and IMU integration.

### Output

```
LaneOffsetResult
    offset           float    normalized [−1.0, +1.0]; 0.0 = centered
    left_x           float?   pixel x of left boundary anchor; None if absent
    right_x          float?   pixel x of right boundary anchor; None if absent
    lane_width_px    float?   pixel distance between anchors; None if not two_boundary
    confidence       float    mean confidence of anchor candidates
    boundary_count   int      total lane_boundary detections this frame
    mode             str      detection mode (see table above)
    frame_id         int
    timestamp        int
```

### Output Object

```
Phase2Output
    detections       list[DetectionObject]
    lane_offset      LaneOffsetResult
    frame_id         int
    timestamp_ms     int
    detection_count  int
```

### Detection List

```
list[DetectionObject]
```

May be empty if no detections occur in a frame.

### Lane Offset

The `lane_offset` field carries the `LaneOffsetResult` from Stage 5. It is
always populated — if no boundary candidates were detected, `mode` will be
`"none"` and `offset` will be `0.0`. Phase 3 must check `mode` before
using `offset` as a steering signal.

### Phase 2 -> Phase 3 Interface

The final output handed to navigation is `Phase2Output`, which provides
structured detections, a lane offset estimate, and frame metadata.

Phase 3 reads `lane_offset.offset` directly as the steering error signal
into the PID controller. Before using the value, Phase 3 checks
`lane_offset.mode` — if `mode` is `"none"` or `"width_rejected"`, the
offset is unreliable and Phase 3 substitutes a dead-reckoned estimate
from the exponential moving average and IMU yaw integration.

Phase 3 then performs temporal filtering, confidence thresholding, and
navigation state estimation.

### Zero 2 W Frame Budget

Operating point: 480×360 @ 20 FPS (50 ms budget)

| Stage                     | Time (ms) |
|---------------------------|-----------|
| Capture + buffer load     | ~1–2      |
| Preprocess                | ~5        |
| ROI crop                  | <1        |
| Color branch              | ~4        |
| Geometry branch           | ~8        |
| Feature fusion            | ~2        |
| Lane offset estimation    | <1        |
| Output packaging          | <1        |
| Phase 3 filtering         | ~3        |
|---------------------------|-----------|
| Total estimate            | ~24 ms    |

Headroom against the 50 ms budget is approximately 26 ms under normal
conditions. Histogram equalization dominates when enabled (~35 ms
additional) and is the highest-impact toggle for recovering frame budget.
The geometry branch (Canny + contour extraction) is the largest fixed cost
at approximately 8 ms and is the correct target if further reduction is
needed without disabling equalization.

---

## Summary

Phase 2 converts a raw camera frame into structured perception outputs using six stages:

1. **Image conditioning** - equalization, blur
2. **Region-of-interest segmentation** - three ROIs for parallel processing
3. **Parallel feature detection** - color branch (HSV) and geometry branch (Canny + contours)
4. **Feature fusion** - conflict resolution and schema normalization
5. **Lane offset estimation** - lateral offset from lane center
6. **Output packaging** — single container handed to Phase 3
