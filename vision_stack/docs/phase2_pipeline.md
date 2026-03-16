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
                                              Perspective Transform
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

#### Lens Undistortion

Removes optical distortion introduced by the camera lens.

```
frame → undistorted_frame
```

#### Histogram Equalization

Equalization is performed in YCrCb space so that intensity contrast improves while color information remains intact.

```
BGR → YCrCb
equalize(Y channel)
YCrCb → BGR
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
color: BGR
```

---

## Stage 2 — ROI Cropping

**File:** `roi_crop.py`

The frame is partitioned into three regions of interest (ROIs) before branching.

### ROI Layout

| ROI | Frame Region | Purpose |
|---|---|---|
| `lane_roi` | lower half of frame | lane boundary detection |
| `traffic_roi` | top-center region | traffic light detection |
| `sign_roi` | right half of frame | stop sign detection |

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
BGR ROI
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
BGR ROI
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

- Normalise detection formats
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

| Class | Rule |
|---|---|
| `traffic_light` | Only one color may exist per frame — keep highest confidence candidate |
| `lane_boundary` | All candidates are kept — multiple lane boundaries are valid |
| `stop_sign` | Only the strongest candidate is kept — keep highest confidence |

---

## Stage 5 — Perspective Transformation

**File:** `perspective_transform.py`

Converts camera perspective coordinates into bird's-eye view coordinates using a homography matrix.

### Why This Is Necessary

In perspective space, parallel lines converge and pixel distances vary with depth. After homography, parallel lines remain parallel and pixel distance becomes uniform. This allows accurate lateral distance measurements.

### Transformation Modes

#### Mode A — Image Warp

```
warpPerspective(lane_roi)
```

Produces a top-down lane image.

```
WarpResult
    warped_image
    homography
    source_shape
```

#### Mode B — Point Transform

Transforms individual points instead of the entire image.

```
transform_points(points)
```

```
PointTransformResult
    transformed_points
    source_points
    homography
```

---

## Stage 6 — Phase 2 Output Packaging

**File:** `phase2_output.py`

Final stage of Phase 2. No computation occurs here. This stage packages all detection outputs into a single container.

### Output Object

```
Phase2Output
    detections
    transformed_coords
    frame_id
    timestamp_ms
    detection_count
```

### Detection List

```
list[DetectionObject]
```

May be empty if no detections occur in a frame.

### Transformed Coordinates

Optional structure containing bird's-eye information.

```
TransformedCoords
    warped_image
    transformed_points
    source_points
    output_width
    output_height
```

At least one of `warped_image` or `transformed_points` must be non-`None`.

---

## Phase 2 → Phase 3 Interface

The final output handed to navigation is `Phase2Output`, which provides structured detections, optional bird's-eye geometry, and frame metadata.

Phase 3 then performs temporal filtering, lane center estimation, and navigation control.

---

## Zero 2 W Frame Budget

| Stage | Time (ms) |
|---|---|
| Capture + buffer load | ~1-2 |
| Preprocess | ~3-5 |
| ROI crop | <1 |
| Color + Geometry branch | ~8-12 (???) |
| Feature fusion | ~1-2 |
| Perspective transform | ~3-5 |
| Output packaging | <1 |
| Phase 3 filtering | ~3-5 |
|---|---|
| Total estimate |~20-26 (at 640x480) |

Summary is ok, but considering thermal throttling under a sustained load, remaining headroom drops
significantly. Consider alternatives to save on frame budget.

## Summary

Phase 2 converts a raw camera frame into structured perception outputs using six stages:

1. **Image conditioning** — undistortion, equalization, blur
2. **Region-of-interest segmentation** — three ROIs for parallel processing
3. **Parallel feature detection** — color branch (HSV) and geometry branch (Canny + contours)
4. **Feature fusion** — conflict resolution and schema normalisation
5. **Perspective transformation** — homography to bird's-eye coordinates
6. **Output packaging** — single container handed to Phase 3

The pipeline is deterministic, modular, and designed for predictable runtime behavior on resource-constrained hardware.
