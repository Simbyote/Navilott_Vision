# Operations Reference

> Calibration, telemetry & logging, and failure mode analysis.

---

## Table of Contents

- [Calibration Procedure](#calibration-procedure)
- [Telemetry and Logging](#telemetry-and-logging)
- [Failure Mode Analysis](#failure-mode-analysis)

---

## Calibration Procedure

> *Calibration must be performed on the actual course hardware under actual course lighting.*

Calibration files should be version-controlled alongside source code. If a threshold change breaks detection, you need to be able to roll back.

### Step 1 — Camera Intrinsic Calibration

Generates the camera matrix and distortion coefficients used by the undistortion stage in Phase 2.

1. Print or display a checkerboard calibration pattern (recommended: 9×6 inner corners)
2. Capture 20–30 images of the pattern at varied angles, distances, and positions across the frame
3. Run OpenCV `calibrateCamera()` to compute the camera matrix and distortion coefficients
4. Save output to `calibration/camera_matrix.npz` — loaded at pipeline startup
5. Re-run if the lens, sensor, or camera mount is physically moved

### Step 2 — HSV Range Tuning

Determines the H/S/V min/max thresholds for traffic light color detection.

1. Capture representative frames under actual course lighting conditions — not lab or office lighting
2. Use an interactive trackbar tool to tune H/S/V min/max ranges for red, yellow, and green independently
3. Verify binary masks isolate only the intended color regions with no bleed from background
4. Save tuned ranges to `calibration/hsv_ranges.json`
5. Document the lighting condition alongside the ranges — indoor fluorescent and outdoor daylight will need different values

### Step 3 — Homography Calibration

Generates the perspective transform matrix used to convert the camera view into a bird's-eye coordinate system.

1. Place a flat grid of known physical dimensions on the course surface within the camera's field of view
2. Identify four corresponding point pairs between the camera image and the target top-down plane
3. Compute the homography matrix using `cv2.getPerspectiveTransform()`
4. Verify that parallel lane lines appear parallel in the transformed output
5. Save to `calibration/homography_matrix.npz`
6. Re-run any time camera angle, mounting height, or tilt changes

### Step 4 — Lane Detection Threshold Verification

1. With all calibration files loaded, run the pipeline on a static course frame
2. Verify Canny thresholds produce clean lane edges without excessive noise or fragmentation
3. Adjust morphological operations (dilation/erosion kernel size) if edges are broken or over-connected
4. Save final Canny thresholds to `calibration/edge_thresholds.json`

### Step 5 — End-to-End Verification

1. Run the full live pipeline on a course walkthrough at operating speed
2. Confirm lateral offset reads near zero when robot is centered in the lane
3. Confirm traffic light state matches ground truth at expected detection distance
4. Confirm stop sign detection triggers consistently at expected range
5. Record baseline confidence values and output jitter — use these as detection stability thresholds in benchmarking

---

## Telemetry and Logging

> *Structured runtime logs are the primary debugging tool*

### Log Format

One row per processed frame, written to a CSV file for offline analysis:

```
timestamp_ms, frame_id, capture_time_ms, preprocess_time_ms, feature_extract_time_ms,
detection_time_ms, fusion_time_ms, phase3_time_ms, total_frame_time_ms,
cpu_percent, mem_usage_mb, lane_offset, heading_error, traffic_state,
detections_count, confidence_mean, confidence_variance, output_jitter_ms
```

### Sample Row

```csv
1712345678123, 00420, 1.2, 3.8, 9.1, 4.3, 1.1, 2.2, 21.7, 58.3, 134, -0.12, 2.4, go, 2, 0.87, 0.03, 1.1
```

### Log Analysis

| Field                 | Warning Threshold    | Likely Cause                                          |
|-----------------------|----------------------|-------------------------------------------------------|
| `total_frame_time_ms` | > 33.3 ms            | Pipeline stage overrun — check per-stage timings      |
| `cpu_percent`         | > 70% sustained      | Contention — consider CPU mitigation options          |
| `mem_usage_mb`        | > 400 MB             | Memory pressure — profile buffer allocations          |
| `confidence_variance` | > 0.15 across frames | Unstable detection — re-tune thresholds or HSV ranges |
| `output_jitter_ms`    | > 5 ms               | Temporal filtering insufficient — widen window        |

### Log Rotation

- Print rolling statistics to stdout every 30 frames during development
- Write full CSV log to disk for post-run analysis
- Flush log buffer on clean exit and on exception to avoid data loss on crash

---

## Failure Mode Analysis

| Failure            | Cause                           | Behavior                        | Mitigation                                                             |
|--------------------|---------------------------------|---------------------------------|------------------------------------------------------------------------|
| Camera feed stall  | libcamera / driver crash        | Perception halts entirely       | Watchdog monitors frame timestamps. Reinitialize `cv2.VideoCapture`    |
| CPU overload       | Heavy frame processing          | Frame drops, timing violations  | Drop to 320×240 or 15 FPS; profile which stage is the bottleneck first |
| Lighting variation | HSV thresholds out of range     | Traffic light misclassification | Pre-tune HSV ranges under actual lighting. Add histogram equalization  |
| Motion blur        | Robot speed too high            | Lane detection unstable         | Increase shutter speed via libcamera exposure controls                 |
| Dropped frames     | appsink drop policy             | Perception jitter               | Reduce frame rate to 15-25 FPS                                         |
| Homography error   | Camera shifted/surface not flat | Incorrect lateral offset        | Re-run homography calibration                                          |
| False positives    | Confidence threshold too low    | Erratic navigation commands     | Raise confidence threshold; widen temporal filter window from N to N+2 |
| UART packet loss   | Noise or baud rate mismatch     | MCU acts on stale data          | Validate checksum on MCU side                                          |
| RAM exhaustion     | Too many simultaneous buffers   | Python OOM crash                | Profile memory early. Enforce in-place OpenCV operations.              |
| TFLite timeout     | Model too large for ROI input   | Latency budget exceeded         | Reduce input resolution to ROI crop only; switch to smaller model      |
