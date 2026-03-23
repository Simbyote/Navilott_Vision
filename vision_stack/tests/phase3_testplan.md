# AutoBot — Perception System

## Phase 3 Unit Tests — Navigation Signal Processing

Tests covering temporal filtering, sensor fusion, UART packet output, and end-to-end frame budget. Maps to benchmark stages 2–7.

---

## Phase 3 — Navigation Signal Processing

### Test 3.1 — Temporal Filtering — Detection Smoothing & Noise Rejection

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 3 — Navigation Signal Processing |
| **Equipment:** | Raspberry Pi Zero 2 W, synthetic `Phase2Output` stream (scripted injection), SSH terminal |
| **Description:** | Verify that the temporal filter smooths detection outputs across frames N, N+1, N+2 and rejects single-frame noise spikes without delaying valid detections. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Inject 10 consecutive frames with `lane_offset = 0.10 m`. Record Phase 3 filtered output. | Filtered `lane_offset` converges to 0.10 m within 3 frames. Steady-state output = 0.10 ± 0.01 m. | | |
| 2 | Inject a single-frame spike: one frame with `lane_offset = 0.90 m` surrounded by frames at 0.10 m. | Spike is attenuated. Filtered output does not exceed 0.30 m during the spike frame. | | |
| 3 | Inject a sustained step change: frames 1–5 at 0.10 m, frames 6–10 at 0.50 m. Verify filter tracks the step. | Filtered output transitions to 0.50 m within 3 frames of step change (no more than N+3). | | |
| 4 | Inject 5 consecutive frames with confidence = 0.10 (below threshold). Verify output is suppressed. | No navigation command issued when confidence < threshold for ≥ 3 consecutive frames. | | |

---

### Test 3.2 — State Estimation — Sensor Fusion (Vision + IMU + Encoder)

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 3 — Navigation Signal Processing |
| **Equipment:** | Raspberry Pi Zero 2 W, IMU module (I2C), encoder wired to GPIO, SSH terminal |
| **Description:** | Verify that state estimation correctly fuses vision-derived coordinates with IMU yaw rate and encoder odometry. Validate that encoder dead-reckoning activates when vision confidence drops below threshold. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | With camera active and lane visible, hold robot stationary over lane center. Record `lane_offset` and encoder `wheel_speed`. | `lane_offset ≈ 0`. `wheel_speed ≈ 0 m/s`. IMU `yaw_rate ≈ 0 deg/s`. | | |
| 2 | Cover the camera (block vision). Force encoder to report `wheel_speed = 0.3 m/s` for 10 frames. Verify dead-reckoning activates. | State estimation switches to encoder dead-reckoning. Navigation output does not go to null/zero. Estimated position updates each frame. | | |
| 3 | With camera blocked, rotate the robot body by ~15°. Verify IMU `yaw_rate` integration propagates `heading_error`. | `heading_error` field increases proportionally to rotation angle. IMU integration bridges the gap. | | |
| 4 | Restore camera. Verify state estimation transitions back to vision-primary after 3 frames of valid high-confidence detection. | `lane_offset` and `heading_error` return to vision-based values within 3 frames. Dead-reckoning flag clears. | | |
| 5 | Verify I2C IMU burst-read time budget: measure time to read `yaw_rate` and `lateral_accel` per frame cycle. | I2C read time < 1 ms per frame cycle. | | |

---

### Test 3.3 — Navigation Packet — UART Output (Option B: Pi + MCU)

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 3 — Navigation Signal Processing (UART Output) |
| **Equipment:** | Raspberry Pi Zero 2 W, MCU (STM32 or Arduino Nano), UART connection (TX/RX + GND), PC with serial monitor |
| **Description:** | Verify that the navigation packet is correctly formatted, transmitted over UART at 115200 baud, and received by the MCU with valid checksum. Covers all four packet types. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Pi transmits a type `0x01` (`lane_offset`) packet with value = -0.12 m. MCU reads and prints received bytes to serial monitor. | MCU receives `[0xAA, 0x01, <4 float bytes for -0.12>, <checksum>, 0xFF]`. Checksum passes. | | |
| 2 | Pi transmits type `0x03` (`traffic_state`) with value = 2 (stop). MCU serial monitor prints decoded traffic state. | MCU decodes `traffic_state = 2`. Serial monitor output: `'STOP'`. | | |
| 3 | Deliberately corrupt the checksum byte before transmission. Verify MCU rejects the packet. | MCU prints 'Checksum error'. Motor state unchanged. MCU does not act on the corrupted packet. | | |
| 4 | Transmit 100 packets in sequence at 115200 baud. Count successful receptions on MCU side. | ≥ 99 of 100 packets received and validated (≤ 1% loss on a clean UART line). | | |
| 5 | Measure UART transmission time for one 7-byte packet at 115200 baud. | Transmission time < 2 ms per packet. | | |

---

### Test 3.4 — End-to-End Frame Budget — Full Pipeline Timing

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 3 — Navigation Signal Processing (End-to-End) |
| **Equipment:** | Raspberry Pi Zero 2 W, IMX219 camera module, full pipeline active (Phases 1–3), telemetry CSV logging active, SSH terminal |
| **Description:** | Verify that the full pipeline from frame capture to validated navigation output completes within the 33.3 ms per-frame budget at 30 FPS under sustained load. Validates the performance budget defined in the architecture documentation. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Run the full pipeline (Phases 1–3) on the live camera stream for 60 seconds. Log `total_frame_time_ms` per frame to CSV. | Mean `total_frame_time_ms` < 27 ms. No single frame exceeds 33.3 ms during the 60 s run. | | |
| 2 | Run the pipeline for 5 minutes (thermal soak). Monitor CPU temperature (`vcgencmd measure_temp`). Log all frame timings. | Frame times do not degrade by more than 20% vs. cold-start baseline after 5 minutes. No thermal throttle above 80°C. | | |
| 3 | Check per-stage timings in the CSV: `preprocess`, `feature_extract`, `detection`, `fusion`, `phase3` fields. | Each stage within documented budget: preprocess < 5 ms, color+geometry < 12 ms, fusion < 2 ms, phase3 < 5 ms. | | |
| 4 | Verify CPU utilization does not exceed 70% sustained (`cpu_percent` field in CSV). | Mean `cpu_percent` < 70% over 60 s run. | | |
| 5 | Check `mem_usage_mb` in CSV. Verify pipeline stays within memory budget. | Peak `mem_usage_mb` < 200 MB (single process). No upward drift over 5 min run. | | |
| 6 | Introduce a real-world stress condition: run a background CPU-intensive shell command (`yes > /dev/null &`) during the 60 s run. Observe frame timing degradation. | Mean frame time rises but ≥ 90% of frames still complete < 33.3 ms. No pipeline crash. | | |
