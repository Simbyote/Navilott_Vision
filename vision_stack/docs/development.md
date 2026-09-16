# Development Reference

> Roadmap and benchmark & testing catalog.

---

## Table of Contents

- [Development Roadmap](#development-roadmap)
- [Benchmark & Testing Reference](#benchmark--testing-reference)

---

## Development Roadmap

```
Stage 1: Camera Capture Validation
  - GStreamer pipeline confirmed working at 480x360 @ ~20 FPS
  - Frame timestamps verified
  - Benchmark: Baseline Single-Stage (capture only)
  - Pass criteria: sustained ~20 FPS, < 10 ms capture time

Stage 2: Preprocessing Benchmarks
  - Histogram equalization + Gaussian blur applied per frame
  - Calibration files loaded from disk
  - Benchmark: Stage-Timing (preprocessing only)
  - Pass criteria: < 5 ms per frame

Stage 3: Lane Detection
  - ROI crop + Canny + contour extraction
  - Lateral offset computed via pixel-based estimation (LaneOffsetResult)
  - Benchmark: Detection Stability (lanes)
  - Pass criteria: stable normalized offset on static course frame; mode != "none" when lane visible

Stage 4: Traffic Light Detection
  - HSV thresholding + blob filtering
  - State classification: red / yellow / green
  - Benchmark: Detection Stability (traffic lights)
  - Pass criteria: correct classification under course lighting

Stage 5: Stop Sign Detection
  - Geometry branch contour matching for octagon shape
  - Benchmark: Detection Stability (stop signs)
  - Pass criteria: consistent trigger at expected range

Stage 6: Feature Fusion + Phase 3 Filtering
  - Both branches merged, temporal filtering active
  - Confidence thresholding tuned
  - Benchmark: Full pipeline CPU & Memory Observation
  - Pass criteria: total frame time < 50 ms, CPU < 70%

Stage 7: Navigation Output
  - Validated packet formatted and transmitted (UART or GPIO)
  - MCU or Pi motor loop receiving and acting on packets
  - Benchmark: Frame-Drop / Real-Time Stress
  - Pass criteria: no timing violations under 60 sec sustained run

Stage 8: System Integration
  - Full pipeline running on course
  - Telemetry logging active
  - End-to-end latency verified
  - Detection stability thresholds set from observed data
```

---

## Benchmark & Testing Reference

### Testing Parameters

| Parameter        | Value / Options                                                           |
|------------------|---------------------------------------------------------------------------|
| Resolution       | 320×240 (stress relief), **480×360** (operating point), 640×480 (upper bound) |
| Frame rate       | 15 FPS (relaxed), **~20 FPS** (operating point), 30 FPS (upper bound)    |
| Print interval   | Every 30 frames                                                           |
| Test duration    | 30 sec (quick), 60 sec (stable), 5–10 min (thermal/long-run)              |

### Benchmark Catalog

| # | Benchmark                     | Goal                                                           |
|---|-------------------------------|----------------------------------------------------------------|
| 1 | Baseline Single-Stage         | Measure computational cost of one processing stage per frame   |
| 2 | Stage-Timing                  | Measure timing cost of one stage inside a live video loop      |
| 3 | Frame-Drop / Real-Time Stress | Stress-test whether a stage can sustain 30 FPS live stream     |
| 4 | CPU & Memory Observation      | Report CPU % and peak memory during single-stage execution     |
| 5 | Detection Stability           | Evaluate output stability (confidence variance, output jitter) |

### Metrics Tracked

| Category              | Metrics                                                                                     |
|-----------------------|---------------------------------------------------------------------------------------------|
| Throughput            | Capture FPS, processing FPS, output FPS                                                     |
| End-to-End Latency    | Frame capture timestamp, post-processing timestamp, final output timestamp                  |
| Frame Drop Behavior   | Total frames grabbed, total processed, estimated dropped                                    |
| CPU Usage             | Average CPU %, peak CPU %                                                                   |
| Memory Usage          | Local memory usage, peak memory usage                                                       |
| Stage-by-Stage Timing | Capture, preprocess, feature extraction, detection, fusion, output packaging                |
| Detection Stability   | Successful detections, false positives, false negatives, confidence variance, output jitter |
