# Phase 3: State Estimation

> Per-frame perception plus sensors to a navigation packet.

Phase 2 answers "what's in this frame". Phase 3 answers "what's true across the last few frames, given the sensors". It smooths the lane offset, bridges short dropouts, tracks heading while vision is lost, and votes on traffic light and stop sign state. The result is one `EstimationPacket` per frame, handed to navigation.

**Code:** `src/estimation.py` · **IMU:** `src/peripherals/imu.py` · **Harness:** `src/phase3_linker.py` · **Tests:** `src/tests/test_estimation.py`, `test_imu.py`

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

The traffic light path is effectively off: the color branch produces nothing until HSV ranges are calibrated, so `drive_state` stays `go`. The stop sign path is live, gated at 0.45 on geometry that hasn't been tuned on course frames yet.

---

## IMU

`IMUReader` samples the MPU-6050 at 0x68 on a background thread at 100 Hz (`IMU_RATE_HZ`), with the on-chip low-pass filter at 44 Hz to stay under the 50 Hz Nyquist limit. The camera frame rate is too slow to integrate the gyro cleanly, so the thread accumulates between frames.

```
calibrate()       average gyro Z at standstill → bias (2 s at 200 samples)
start()           background thread, paced to 100 Hz; read errors counted, not fatal
snapshot()        once per frame: swap out the accumulator → IMUFrame
stop()
```

`snapshot()` holds the lock only long enough to swap accumulators, so the cost is the same whatever the frame rate. Each `IMUFrame` carries the mean bias-corrected yaw rate, the signed lateral acceleration with the largest magnitude, and a sample count; it's invalid if no sample arrived.

`SensorSample.from_imu()` converts it for Phase 3. An invalid frame gives `None` readings, which the heading tracker treats as "hold".

If `calibrate()` is used, leave `Phase3Config.gyro_bias_dps` at 0; the bias is already subtracted. `phase3_linker --imu` doesn't calibrate, so pass `--gyro-bias` there instead.

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
| `yaw_rate` | `float` | Pass-through, deg/s; 0.0 if unavailable |
| `lateral_accel` | `float` | Pass-through, m/s²; 0.0 if unavailable |
| `wheel_speed` | `float` | Pass-through, m/s; 0.0 until encoders are wired in |
| `frame_id`, `timestamp_ms` | `int` | The frame's stamp, carried from capture |

A pass-through of 0.0 can mean "zero" or "unavailable". If navigation needs to tell them apart, that's a contract change to agree on.

The age of a packet is `now − timestamp_ms` on the same monotonic clock (`time.monotonic_ns() // 1_000_000`). What navigation does with a stale packet or one that's too old is its decision, but it should be written down alongside this table.

---

## Sign conventions

| Quantity | + means | Source |
| --- | --- | --- |
| `lane_offset` | Robot right of lane center | `lane_offset.py` |
| `heading_error`, `yaw_rate` (per `estimation.py`) | Turning right | `estimation.py` |
| Gyro Z (per `imu.py`) | Counter-clockwise, turning **left**, for a flat Z-up mount | `imu.py` |
| Accel Y (per `imu.py`) | Accelerating left | `imu.py` |

**These disagree.** `imu.py` documents + yaw as turning left, `estimation.py` documents it as turning right, and `SensorSample.from_imu()` passes the value through unchanged. Either the IMU is mounted so the axis is flipped, or `heading_error` has the wrong sign. Check on the bench: rotate the robot right by hand with `phase3_linker --camera --imu` running, and see which way `yaw_rate` goes. Then negate in `from_imu()` if needed and fix whichever comment is wrong.

---

## Running it

`phase3_linker` runs capture, Phase 2 and Phase 3 headless and reports each packet next to the Phase 2 input it came from, so a bad packet can be traced to bad input or bad filtering.

```
python3 -m src.phase3_linker --video run.avi
python3 -m src.phase3_linker --camera --imu --fps 20
python3 -m src.phase3_linker --camera --limit 200 --print-every 1
```

| Output (`runs/p3_<timestamp>/`) | Contents |
| --- | --- |
| Console | A status line once a second, plus an event line whenever `lane_status`, `drive_state` or `stop_sign_detected` changes |
| `p3.csv` | Every frame: timings, the Phase 2 lane input, the packet, and Phase 3's log |
| `summary.txt` | Timing percentiles, frames over budget, lane status and mode histograms, longest hold and stale runs, offset statistics while on vision |

Replays are deterministic, so Phase 3 config changes can be compared on the same footage. With the robot parked centered, the offset standard deviation in `summary.txt` is the measurement noise floor.

---

## Not in Phase 3 (yet)

- **Dead reckoning during a hold.** The hold repeats the last offset. Once wheel encoders are wired in, odometry plus heading could propagate it instead (the `@TODO` in `LaneFilter.update()`).
- **Scene awareness.** Intersections, turns and orientation are not modeled; there's only lane offset. The planned scene state machine goes here, as a separate entry point so the current path keeps working.
- **Out-of-bounds recovery** (stop, localize, correct). Deferred until basic lane keeping works.
- **Control.** Phase 3 ends at the packet. Steering and speed are navigation's.

---

## Open items

- **IMU yaw sign**, as above, before anyone uses `heading_error`.
- **Stop-sign gate.** The stop sign reaches the vote at 0.45 on untuned geometry. Keep the gate high, or have navigation ignore `stop_sign_detected`, until the sign branch is tuned on course frames.
- **Hold length.** 7 frames is a starting value. Check the longest hold and stale runs in `summary.txt` from the course recordings.
- **`cm_per_px`** measured against the known lane width (about 14 cm), so `lane_offset_cm` can be checked against the ±2 cm requirement.
- **The navigation contract.** Agree with navigation on the stale-packet behavior and the pass-through ambiguity above, and record it here.
