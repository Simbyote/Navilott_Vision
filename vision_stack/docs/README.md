# Navilott Documentation

Navilott is an autonomous robot that drives a street-style course on its own: it follows its lane, stops at stop signs and obeys traffic lights. A camera and a classical computer-vision pipeline, running entirely on a Raspberry Pi Zero 2 W, turn what the robot sees into a lane offset and traffic signals that the navigation code steers by.

Senior Design Team 2.08 Magnetronics, Texas State University.

---

## Start here

| If you want to… | Read |
| --- | --- |
| Understand the whole system | `architecture.md`, then `vision_stack/` in order |
| Know what the system must do, and what's proven so far | `requirements.md` |
| Know the course and camera measurements the tuning depends on | `course.md` |
| Consume the vision output in navigation code | `vision_stack/phase3_estimation.md`, "Contract" and "Sign conventions" |
| Run the tests | `guides/pytest.md` |
| Watch what the pipeline detects | `guides/phase2_linker.md` |
| Check what estimation decides | `guides/phase3_linker.md` |
| Calibrate the camera lens | `guides/calibrate_camera.md` |
| Check an existing calibration | `guides/test_calibration.md` |

---

## Layout

```
docs/
    vision_stack/
        phase1_capture.md       camera → timestamped frames
        phase2_perception.md    frame → detections and lane offset
        phase3_estimation.md    detections + IMU → navigation packet
    guides/
        calibrate_camera.md     lens calibration
        phase2_linker.md        watching the pipeline, with video
        phase3_linker.md        checking estimation, headless
        pytest.md               running the test suite
        test_calibration.md     verifying a calibration
    architecture.md             hardware, wiring, power, software layout, risks
    course.md                   course dimensions and camera geometry
    README.md                   this page
    requirements.md             requirements, how each is checked, status
```

Navigation (driving, route logic, encoders). **TBD**

---

## System Overview

```
IMX290 camera ──► Phase 1: capture      one frame, id and timestamp
                     │
                     ▼
                  Phase 2: perception   lane boundaries, lane offset,
                     │                  stop sign, traffic light (per frame)
                     ▼
MPU-6050 IMU ──►  Phase 3: estimation   smoothed offset, hold/stale status,
                     │                  heading, voted drive state
                     ▼
                  EstimationPacket
                     │
                     ▼
                  Navigation ──► TB6612FNG ──► N20 motors
```

---

## Conventions used everywhere

| | Convention |
| --- | --- |
| Frame size | 480 × 270, BGR, upright |
| Frame identity | Every result carries the `frame_id` and `timestamp_ms` capture assigned; nothing else creates them |
| Time | Monotonic milliseconds, meaningful only as differences within one run |
| Lane offset | [−1, +1] of half the lane ROI width. **+ = robot right of lane center, so steer left** |
| Lane status | Steer only on `vision` or `hold`; `stale` means don't trust the offset |
| Constants | Shared values live in `src/params.py`; each stage's tuning lives in its config, bundled in `PipelineConfig` |
| Commands | Repo commands run from `~/Navilott_Vision`; pytest runs from `~/Navilott_Vision/vision_stack` |

---

## Current status

The pipeline runs end to end and the software tests pass. Nothing has been verified on the course with the IMX290 camera yet. `requirements.md` has the per-requirement status, and each `vision_stack/` doc ends with its open items. The most important:

- Confirm the lane offset sign and the IMU yaw sign on the robot before tuning
- Measure the ground scale, then check lane offset accuracy against ±2 cm
- Add a motor watchdog before running at speed
