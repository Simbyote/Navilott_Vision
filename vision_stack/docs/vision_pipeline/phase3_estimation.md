# Phase 3: State Estimation

> Per-frame perception plus sensors to a navigation packet.

Phase 2 answers "what's in this frame". Phase 3 answers "what's true across the last few frames, given the sensors". It smooths the lane offset, bridges short dropouts, tracks heading while vision is lost, and votes on traffic light and stop sign state. The result is one `EstimationPacket` per frame, handed to navigation.

**Code:** `src/estimation/estimation.py` · **Sensors:** `src/peripherals/sensing.py` over `src/peripherals/imu.py` and `drive.py` · **Harness:** `src/phase3_linker.py`, with `src/debugger/estimation_debug.py` (the traced twin) and `src/debugger/debug_phase3.py` (the video) · **Tests:** `src/tests/test_estimation.py`, `test_sensing.py`, `test_imu.py`

---

## Design philosophy

**The line between Phase 2 and Phase 3 is memory.** A stage belongs in Phase 3 only if it uses history (smoothing, voting, holding) or sensor data. Per-frame judgments about pixels belong in Phase 2. That keeps Phase 2 testable one frame at a time and puts all state in one place.

**Each stage owns only its own state.** The lane filter, heading tracker and the two classifiers are separate small classes. None can read or change another's state, so each can be tested alone with a sequence of hand-built inputs.

**The order is written in one place.** `Phase3Processor.process()` is the only code that runs the stages in sequence, the same rule `run_chain()` follows for Phase 2.

**It runs without the robot.** `estimation.py` never imports the IMU driver, which loads board libraries at import time. The IMU is passed in as a plain `SensorSample`, so estimation runs and tests on a laptop, from replays, with no sensors at all.

**It never invents a measurement.** When vision drops out, the offset is held and marked as held, then marked stale. Navigation is told how much to trust the number rather than handed a guess that looks like a measurement.

---

## Stage order

```
Phase2Output + SensorSample
   │
   ├─ dt since previous frame             clamped to [0, 0.5 s]
   ▼
LaneFilter          jump gate → EMA → hold / stale        → lane_offset, lane_status
   │
   ▼
HeadingTracker      integrate yaw while not on vision     → heading_error
   │
   ▼
TrafficClassifier   confidence gate → 3-frame vote        → drive_state
   │
   ▼
StopSignClassifier  confidence gate → 3-frame vote        → stop_sign_detected
   │
   ▼
StopLineClassifier  3-frame vote → hold distance          → stop_line_detected, stop_line_distance_px
   │
   ▼
EstimationPacket    → Navigation
```

Create one `Phase3Processor` per run and call `process()` on every frame, in order. It's stateful; a new processor starts from nothing.

---

## Lane filter

Turns the per-frame `LaneOffsetResult` into a smoothed offset and a status.

A frame counts as a **measurement** only if:

1. its mode is `two_boundary`, `left_only` or `right_only` (`none` and `single_uncalibrated` report 0.0 with nothing behind it), and
2. its offset is within `max_offset_jump` (0.5) of the current estimate. A larger change between two frames isn't physically plausible at driving speed, so it's treated as a dropout.

| Status | When | Offset reported |
| --- | --- | --- |
| `vision` | This frame was a measurement | EMA of the measurements |
| `hold` | Dropout, for up to `hold_max_frames` (7, about 350 ms at 20 FPS) | The last good value |
| `stale` | Dropout for longer than that | The last good value, which navigation must not steer by |

- **EMA:** `α = 0.35`. Higher follows faster and smooths less. The first measurement seeds it directly.
- **Stale resets the EMA,** so the next usable frame re-seeds the estimate. Otherwise a robot that moved during a long dropout would have every new measurement rejected by the jump gate, forever.
- **Before the first measurement** the status is `stale` and the offset is 0.0.
- **Centimeters:** with `cm_per_px` and `lane_roi_width_px` set, `lane_offset_cm` undoes the normalization. Until a scale is measured it's `None`. `phase3_linker --cm-per-px` sets a hand-measured value.

The hold only repeats the last value. It doesn't dead-reckon from the IMU or encoders yet; see Open items.

---

## Heading tracker

While the lane filter is on `vision`, the offset already carries any correction, so heading resets to 0. Once vision drops, the tracker integrates gyro yaw rate (minus `gyro_bias_dps`) over each frame's `dt`, clamped to ±90°.

The result is **how far the robot has turned since the last good frame**, not an absolute heading relative to the lane. That's what a recovery maneuver needs: "I've turned 20° right since I lost the lines".

- No yaw rate this frame: heading holds and it's logged
- `dt` over 0.5 s (a stall) is clamped, so one late frame can't integrate a large step

---

## Traffic light and stop sign

Both use the same pattern: a confidence gate per frame, then a majority vote.

| | Traffic light | Stop sign |
| --- | --- | --- |
| Gate | `min_confidence_traffic` 0.40 | `min_confidence_sign` 0.45 |
| Per-frame value | red → `stop`, yellow → `caution`, green → `go`; no gated light → `go` | gated sign present → `True` |
| Output | `drive_state` | `stop_sign_detected` |

**The vote needs a strict majority of the full window,** not of the frames seen so far. With a window of 3, the state changes only when 2 of the last 3 frames agree; otherwise the previous state stands. One frame at startup can't flip it, and a three-way split doesn't pick a winner at random.

Fusion already keeps at most one light and one sign per frame, so there's no candidate selection here.

The traffic light path runs with `calibration/hsv_ranges.json`, loaded whether or not it has been tuned under course lighting. The stop sign path is live, gated at 0.45 on geometry that hasn't been tuned on course frames yet. On synthetic frames every detected sign scores at least 0.53, so the gate is never reached from below there.

## Stop line

`StopLineClassifier` votes on Phase 2's `stop_line_results` with the same `vote_window`. There is no confidence gate here: `stop_line_distance` already refuses candidates under its own `min_confidence`.

| Voted | This frame saw a line | `stop_line_distance_px` |
| --- | --- | --- |
| yes | yes | this frame's distance |
| yes | no | the last measured distances (px and cm, from one frame), held (logged as `[STOPLINE]`) |
| no | either | `None` (px and cm) |

The hold lasts only as long as the vote: with a window of 3, one missed frame keeps the line and its last distance; two in a row drop it. The distance is in lane-ROI px from the bottom of the lane ROI (0 = on the line).

---

## Sensors

The IMU and wheel encoders run faster than the camera (100 Hz against ~20 FPS), so `src/peripherals/sensing.py` collects them between frames and hands each frame one group. `SensorHub` reads both together on one background thread at 100 Hz (`SENSOR_RATE_HZ`), stamps each `SensorReading` on `time.monotonic` (the camera's clock), and keeps up to 2 s of them (`SENSOR_HISTORY_S`; older ones are dropped and counted).

```
SensorHub.open(imu=, encoders=)   open only the drivers asked for
start()     the first window's starting counts; background thread at 100 Hz
drain()     once per frame: close the window with a fresh encoder reading
            (no I2C in the frame loop) → SensorBatch of every reading since
            the last drain
stop()      end the thread; release the encoders and pigpio
```

Each reading is one `IMUReader.read()` (bias-corrected gyro Z and accel Y) and one `EncoderReader.counts()` at the same instant. **Yaw is flipped into Estimation's + = turning right here, once**, by `IMU_YAW_SIGN` in `params.py` (+1 on this robot: its IMU is upside down, so the driver's + = left already reads + = right; set 2026-10-01 once the motor sides were fixed), so nothing downstream knows how the IMU is mounted. A failed IMU read leaves that reading's IMU fields empty and is counted, never raised.

A `SensorBatch` gives Phase 3 what it needs: the mean yaw rate over its IMU readings, the signed lateral acceleration with the largest magnitude, and each wheel's counts per second from the previous batch's last reading to this one's (0.0 stopped). It also carries the cumulative counts, which the packet leaves out. `SensorSample.from_batch()` reads those by name, so `estimation.py` never imports `sensing.py` or the drivers, and it still runs on a laptop from replays with no sensors.

The MPU-6050 is at 0x68, with the on-chip low-pass filter at 44 Hz to stay under the 50 Hz Nyquist limit of 100 Hz sampling. The hub doesn't calibrate it, so `Phase3Config.gyro_bias_dps` (`phase3_linker --gyro-bias`) carries the bias, **in the hub's frame** (after `IMU_YAW_SIGN`; with +1 that's the raw reading at rest). The hub is the only way the IMU and encoders are read: the pipeline, the linkers and `test_imu`'s bench characterization all go through it, and the drivers only answer `read()` / `counts()`.

Using the encoder readings (checking that a steering correction actually turned the wheels, speed control, distance travelled) is Navigation's job, since only Navigation knows what was commanded.

---

## Contract: `EstimationPacket`

| Field | Type | Meaning |
| --- | --- | --- |
| `lane_offset` | `float` | [−1, +1] of half the lane ROI width. **+ = robot right of lane center, so steer left** |
| `lane_offset_cm` | `float \| None` | Same in cm; `None` until a scale is set |
| `lane_status` | `str` | `vision` / `hold` / `stale`. **Steer only on `vision` or `hold`** |
| `heading_error` | `float` | Degrees turned since the last `vision` frame; 0.0 on vision |
| `drive_state` | `str` | `go` / `caution` / `stop`, voted |
| `stop_sign_detected` | `bool` | Voted |
| `stop_line_detected` | `bool` | Voted |
| `stop_line_distance_px` | `float \| None` | Lane-ROI rows from the nearest stop line to the ROI bottom (0 = on it), held through a missed frame; `None` unless `stop_line_detected` |
| `stop_line_distance_cm` | `float \| None` | Floor cm forward of the reference point (the bottom of the camera's view) to where the line crosses the robot's centerline, held with the px value from the same frame; from the stop-line table (cm ahead of where its marks were measured from) when there's no ground homography; also `None` with neither |
| `yaw_rate` | `float` | Pass-through, deg/s; 0.0 if unavailable |
| `lateral_accel` | `float` | Pass-through, m/s²; 0.0 if unavailable |
| `wheel_speed` | `float` | m/s; always 0.0 for now. **To fill in:** needs the encoder counts per wheel revolution and the wheel diameter to convert counts to meters |
| `frame_id`, `timestamp_ms` | `int` | The frame's stamp, carried from capture |
| `left_wheel_cps`, `right_wheel_cps` | `float` | Pass-through: each wheel's encoder counts per second over the frame window, from the sensor hub's `SensorBatch` over `peripherals/drive.py`'s `EncoderReader.counts()`; + = forward. Raw counts, not converted to distance. 0.0 when the wheel is stopped or without encoders; the drivers' presence checks say whether they're connected |
| `lane_mode` | `str` | Pass-through: this frame's Phase 2 lane offset mode (`two_boundary`, `left_only`, `right_only`, `single_uncalibrated`, `none`), unfiltered; `none` when Phase 2 gave no lane result. Navigation ends an intersection crossing on `two_boundary` |

What Navigation must do with each field, and what it returns, is `navigation_contract.md`.

A pass-through of 0.0 can mean "zero" or "unavailable". If navigation needs to tell them apart, that's a contract change to agree on.

The age of a packet is `now − timestamp_ms` on the same monotonic clock (`time.monotonic_ns() // 1_000_000`). What navigation does with a stale packet or one that's too old is its decision, but it should be written down alongside this table.

---

## Sign conventions

| Quantity | + means | Source |
| --- | --- | --- |
| `lane_offset` | Robot right of lane center | `lane_offset.py` |
| `heading_error`, `yaw_rate` | Turning right | `estimation.py`, applied by `sensing.py` |
| Raw gyro Z (per `imu.py`) | Counter-clockwise, turning **left**, for a flat Z-up mount. **On this robot the raw reading is + for a left turn** (measured) | `imu.py` |
| Accel Y (per `imu.py`) | Accelerating left | `imu.py` |

**Resolved (2026-09-30).** The raw gyro on this robot reads + for a left turn (`maneuver_linker`'s spin pulses measure it every run). `SensorHub` multiplies it by `IMU_YAW_SIGN = -1`, so `yaw_rate` and `heading_error` read + = turning right as `estimation.py` documents. `IMU_YAW_SIGN` is per robot: a different mount needs its own measurement. To check on the bench: `phase3_linker --camera --imu`, cover the lens so vision drops to `hold`, turn the robot right by hand, and `hd` should go positive.

---

## Running it

The robot's Phase 3 tuning is `MEASURED_ESTIMATION` in `src/config.py`, beside the Phase 2 `MEASURED`; it holds `Phase3Config`'s defaults until course runs tune it. The main pipeline (`src/pipeline.py`) runs Phase 3 as one stage, `Pipeline.estimate()`, with one `Phase3Processor` built when the `Pipeline` is created. When `cm_per_px` is set, the pipeline takes the lane ROI width from the config's lane ROI at 480×270 (`with_lane_roi_width()`, which `phase3_linker` uses too) and refuses frames of another size. Navigation follows as the pipeline's last stage (`Pipeline.navigate()`, `navigation_contract.md`). `test_pipeline` holds its packets to `phase3_linker`'s over a drive sequence, frame by frame.

`phase3_linker` is the debug pipeline: it runs capture, Phase 2 (`run_chain`) and Phase 3 through `TracedPhase3Processor` (`src/debugger/estimation_debug.py`) and reports each packet next to the Phase 2 input it came from, so a bad packet can be traced to bad input or bad filtering. The traced processor subclasses each production stage and calls its `update()`, so the filter and vote math exist once; it records each decision (lane: raw, reason, EMA before and after, missed count; votes: each detection's confidence against its gate, the buffer and state; stop line: seen or held; heading: reset and step) and times each stage. `pipeline.py` never imports it, and `test_estimation_debug` holds its packets to `Phase3Processor`'s. The records drive the Phase 3 video (`src/debugger/debug_phase3.py`).

```
python3 -m src.phase3_linker --video run.avi
python3 -m src.phase3_linker --camera --imu --fps 20
python3 -m src.phase3_linker --camera --limit 200 --print-every 1
```

| Output (`runs/p3_<timestamp>/`) | Contents |
| --- | --- |
| Console | A status line once a second, plus an event line whenever `lane_status`, `drive_state`, `stop_sign_detected` or `stop_line_detected` changes |
| `p3.csv` | Every frame: timings, the Phase 2 lane input and stop-line distance (`p2_stop_line_px`), the packet (`stop_line_detected`, `stop_line_distance_px` included), and Phase 3's log |
| `p3_debug.avi` / `.csv` | The Phase 3 video and its per-frame decisions (on by default; `--no-video`); see `guides/phase3_linker.md` |
| `summary.txt` | Timing percentiles, frames over budget, lane status and mode histograms, longest hold and stale runs, offset statistics while on vision, per-stage timing (Phase 2, Phase 3, render), the `[PHASE 3]` decision counts |

Replays are deterministic, so Phase 3 config changes can be compared on the same footage. With the robot parked centered, the offset standard deviation in `summary.txt` is the measurement noise floor.

---

## Not in Phase 3 (yet)

- **Dead reckoning during a hold.** The hold repeats the last offset. Once wheel encoders are wired in, odometry plus heading could propagate it instead (the `@TODO` in `LaneFilter.update()`).
- **Scene awareness.** Stop lines are voted and measured, but intersections, turns and orientation are not modeled as states. Where the scene state machine lives (estimation or navigation) is still open.
- **Out-of-bounds recovery** (stop, localize, correct). Deferred until basic lane keeping works.
- **Control.** Phase 3 ends at the packet. Steering and speed are navigation's.

---

## Open items

- **Stop-sign gate.** The stop sign reaches the vote at 0.45 on untuned geometry. Keep the gate high, or have navigation ignore `stop_sign_detected`, until the sign branch is tuned on course frames.
- **Hold length.** 7 frames is a starting value. Check the longest hold and stale runs in `summary.txt` from the course recordings.
- **`cm_per_px`** measured against the known lane width (about 14 cm), so `lane_offset_cm` can be checked against the ±2 cm requirement.
- **The navigation contract** is `navigation_contract.md`. Still to agree there: stale-packet behavior, the pass-through ambiguity above, and whether the packet carries cumulative encoder counts.
