# AutoBot — Perception System

## Phase 1 Unit Tests — Camera Acquisition

Tests covering frame delivery, ring-buffer drop policy, and timestamp integrity. Maps to benchmark stages 1, 2, 6, and 7.

---

## Phase 1 — Camera Acquisition

### Test 1.1 — GStreamer Pipeline Initialization & Frame Delivery

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 1 — Camera Acquisition |
| **Equipment:** | Raspberry Pi Zero 2 W, IMX219 camera module (CSI), PC/laptop with SSH or serial monitor |
| **Description:** | Verify that the GStreamer pipeline initializes correctly and delivers frames at 640×480 @ 30 FPS to the OpenCV layer. Confirms the full camera stack from libcamera through appsink is operational. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | SSH into the Pi. Launch the capture-only benchmark script at 640×480 @ 30 FPS. Print capture timestamp per frame. | `cv2.VideoCapture()` opens without error. Frame timestamps increment each iteration. | | |
| 2 | Run for 30 seconds. Record frames received vs. expected (30 FPS × 30 s = 900 frames). | ≥ 890 frames received (≤ 1% drop under idle conditions). | | |
| 3 | Print capture latency per frame. Compute mean capture time. | Mean capture time < 10 ms per frame. | | |
| 4 | Verify frame shape returned by `cap.read()`. | Frame shape = `(480, 640, 3)`, dtype = `uint8`, color order BGR. | | |
| 5 | Observe console output. Confirm no GStreamer pipeline errors or warnings in stderr. | No pipeline errors. No 'dropped buffer' or 'timeout' messages. | | |

---

### Test 1.2 — Ring Buffer Drop Policy Verification

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 1 — Camera Acquisition |
| **Equipment:** | Raspberry Pi Zero 2 W, IMX219 camera module, PC/laptop with SSH |
| **Description:** | Confirm that `appsink drop=true max-buffers=1` discards stale frames when the pipeline is artificially stalled. Ensures the system always processes the most recent frame rather than queuing latency. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Modify the capture loop to insert a deliberate 100 ms sleep after every frame read to simulate a slow consumer. | Pipeline does not crash. No backpressure error. | | |
| 2 | After 10 seconds, remove the sleep and resume normal capture. Record frame timestamps. | Frame timestamps jump forward by ~100 ms per stall cycle, confirming old frames were dropped rather than queued. | | |
| 3 | Verify frame count during stall: at 100 ms consumer delay + 33 ms capture, effective FPS should be ~8–9. | Processed FPS drops to ~8–10 during stall. No GStreamer buffer overflow. | | |
| 4 | Resume normal capture. Verify system returns to ≥ 28 FPS within 5 frames. | Frame rate recovers to ≥ 28 FPS within 5 frames of sleep removal. | | |

---

### Test 1.3 — Frame Timestamp Integrity

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 1 — Camera Acquisition |
| **Equipment:** | Raspberry Pi Zero 2 W, IMX219 camera module, PC/laptop with SSH |
| **Description:** | Verify that frame timestamps assigned at acquisition are monotonically increasing and consistent with expected inter-frame intervals. Timestamp integrity is critical for Phase 3 temporal filtering. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Capture 300 frames. Record `timestamp_ms` for each frame. | All timestamps monotonically increasing. No duplicates. | | |
| 2 | Compute delta between consecutive timestamps. Expected delta ≈ 33.3 ms at 30 FPS. | Mean delta = 30–37 ms. No deltas > 66 ms (2-frame gap) under idle conditions. | | |
| 3 | Induce a 1-second OS sleep (`sudo sleep 1`) on a separate terminal during capture. Observe timestamp delta. | Timestamps resume after the stall; large delta observed at stall point. Timestamps do not reset to zero. | | |
| 4 | Confirm ring buffer `frame_id` field increments by 1 per frame. | `frame_id` increments by 1 on every `cap.read()` call. | | |

---
