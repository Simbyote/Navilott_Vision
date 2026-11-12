# Navilott Documentation

Navilott is an autonomous robot that drives a street-style course on its own: it follows its lane, stops at stop signs and obeys traffic lights. A camera and a classical computer-vision pipeline, running entirely on a Raspberry Pi Zero 2 W, turn what the robot sees into a lane offset and traffic signals that the navigation code steers by.

Senior Design Team 2.08 Magnetronics, Texas State University.

---

## Start here

| If you want to… | Read |
| --- | --- |
| Understand the whole system: hardware, wiring, power, software layers | `architecture.md`, then `vision_pipeline/` in order |
| Look up a term (Otsu, the GIL, HSV, a homography...) | `concepts.md` |
| Know what the system must do, how each requirement is checked, and what's proven | `requirements.md` |
| Know the course and camera measurements the tuning depends on | `course.md` |
| Follow a frame through capture, perception and estimation | `vision_pipeline/phase1_capture.md`, `phase2_perception.md`, `phase3_estimation.md` |
| Know what Navigation receives, what it returns, and how its rules decide | `vision_pipeline/navigation_contract.md` |
| Run anything | `make` in `vision_stack/` lists every command (below) |

How to use a tool is in the tool itself: each module's header says what it is for, and `--help` on every script, linker and analysis says how to run it.

---

## Layout

```
docs/
    README.md                   this page
    architecture.md             hardware, wiring, power, software layers, risks
    concepts.md                 the ideas the docs use, explained once
    requirements.md             requirements, how each is checked, status
    course.md                   course dimensions and camera geometry
    resolution_plan.md          the review of 12 Nov: each requirement's gap, the fixes, the phases
    phase1_robot_procedure.md   the robot steps that close Phase 1 (lane offset, lane keeping)
    vision_pipeline/
        phase1_capture.md       camera → timestamped frames
        phase2_perception.md    frame → detections and lane offset
        phase3_estimation.md    detections + sensors → navigation packet
        navigation_contract.md  packet in, motor command out: Phase 4, the navigation rules
```

Code lives in `src/`, layered as `architecture.md` ("Software layers") describes; the tests are in `src/tests/`.

---

## Running it

Everything runs from `~/Navilott_Vision/vision_stack`, where `src` can be imported. `make` (or `make help`) prints every command with a one-line description, grouped:

| Group | Commands | What for |
| --- | --- | --- |
| Start here | `make session TIME="YYYY-MM-DD HH:MM"`, `make pigpiod` | Every boot on the Pi: the clock (NTP is blocked), the GPIO daemon, the venv |
| Race day | `make run` | The course: route, start button, countdown, time on the display; nothing recorded (`src/main.py`) |
| Linkers | `make phase2`, `phase3`, `maneuver`, `navigate`, `nav-dry`, `intersection`, `render` | The chain up to one point, recorded with debug video: Phase 2 views, Phase 3 packets, a drive trial, the whole chain driving, one intersection judged |
| Camera | `make camera-controls`, `make sweep-camera` | What the camera's controls are, and each candidate setting scored live on the sign, light and lane; `CAM="..."` sets controls for one run of the linkers and camera diagnostics |
| Routines | `make routines`, `make routine-<name>` | Prompted hardware tests judged PASS / FAIL against a requirement (`src/routines/`) |
| Diagnostics | `make diag-nav`, `diag-run`, `capture-anatomy`, `i2c-trace`, `sched-latency` | How a run uses the Pi: threads, cores, heat, the camera path, the I²C bus, wake-up latency |
| Analysis | `make nav-report`, `loop-budget`, `light-report`, `timing`, `timeline`, `jitter`, `pi-load`, `load-compare`, `camera-meta`, `capture-report`, `sched-report`, `fig8-report`, `motor-health`, `eval`, `label`, `obstacle-map`, `range-compare`, `line-replay`, `approach-log`, `light-clips`, `sign-check`, `compare`... | Findings and figures from a run's recordings; `RUN=` picks the run, default the newest; `BASE=` sets an earlier run beside it where a tool compares |
| Tests | `make test`, `make test-hw`, `make soak`, `make lint` | Software tests on any machine; hardware characterization and the soak on the Pi |
| Calibration | `make calib-camera`, `calib-lane`, `calib-stopline`, `calib-odometry`, `calib-ground`, `calib-lamps`, `sweep-lamps` | The lens, where a centered robot sees its lane and its width in px (for `config.py`), stop-line cm, encoder cm per count, the floor homography, the traffic-light bands; the others write to `calibration/` |

Options are make variables: `VIDEO=` or `FRAMES=` replays instead of the camera, `RUN=` picks a run folder, `ARGS="..."` passes anything else through. Run folders land in `runs/`, hardware-test output in `artifacts/`.

---

## System overview

```
IMX290 camera ──► Phase 1: capture      one frame, id and timestamp
                     │
                     ▼
                  Phase 2: perception   lane boundaries, lane offset, stop line,
                     │                  stop sign, traffic light, obstacle (per frame)
                     ▼
MPU-6050 IMU  ──► Phase 3: estimation   smoothed offset, hold/stale status,
wheel encoders       │                  heading, votes, odometry
                     ▼
                  EstimationPacket
                     │
                     ▼
                  Phase 4: navigation ──► TB6612FNG ──► N20 motors
                  (route, rules, lane keeping)
```

`src/pipeline.py` runs one frame through all four, `step(frame, sensors) -> Command`; `src/main.py` is the loop around it.

---

## Conventions used everywhere

| | Convention |
| --- | --- |
| Frame size | 480 × 270, BGR, upright |
| Frame identity | Every result carries the `frame_id` and `timestamp_ms` capture assigned; nothing else creates them |
| Time | Monotonic milliseconds, meaningful only as differences within one run |
| Lane offset | [−1, +1] of half the lane ROI width. **+ = robot right of lane center, so steer left** |
| Lane status | Steer only on `vision` or `hold`; `stale` means don't trust the offset |
| Heading and yaw | **+ = turning right** |
| Constants | Hardware facts and run-wide switches in `src/params.py`; each stage's tuning in its config class, assembled into `MEASURED` and `MEASURED_ESTIMATION` in `src/config.py`; navigation's constants at the top of each rule in `src/navigation/` |

---

## Current status

The whole chain runs on the robot, from the start button to the finish: lane keeping, stop lines, stop signs, traffic lights, obstacles, intersections and the route. The software tests pass. Every requirement has been verified against its pass rule (bench runs 2–4 Nov, the course run 4 Nov) except D1, whose target is still pending; `requirements.md` has each result, and each `vision_pipeline/` doc ends with its open items. What the results leave to work on:

- The frame rate and latency while driving: 19.2 FPS and 52.2 ms p95 against 20 FPS and 50 ms (P1, P2, course run 2026-11-04); Phase 2's geometry step is the main thread's cost
- The lane offset reads about 63% of the true offset (P3, slope 0.63): the pixel-to-cm scale
- Stop-sign false stops beside a red traffic lamp, 3 in 9.3 min (D2), and lit yellow missed facing a window (D3): `make routine-detect-range`
- The stop-line distance reads about 1.7 cm long (D4b): the stop-line table
- Dark obstacles aren't seen (D6), a known limitation of the brightness and color test: `make routine-obstacle-check` per object class
- D1's pass rule waits on the team: `requirements.md`, Open decisions
