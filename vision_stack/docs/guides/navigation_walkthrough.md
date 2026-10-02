# Navigation Walkthrough

> How one camera frame becomes a motor command, which file decides what, and where to change things. Start here before editing navigation.

The code is split into many small files on purpose: each one owns one decision, so it can be tested alone and tuned without touching the rest. The cost is that the path through them is hard to see. This page is that path.

---

## 1. One frame, end to end

Every frame (about 20 per second) goes down this chain once. Each arrow is a function call; each box is one file.

```
camera frame
   │  src/capture/camera.py           CameraSource.read() -> frame, frame_id, timestamp_ms
   ▼
Pipeline.step()                       src/pipeline.py: calls the three phases below, nothing else
   │
   ├─ perceive()   Phase 2            src/perception/*.py: lane lines, stop line, sign, light in THIS frame
   │                                  -> Phase2Output
   ├─ estimate()   Phase 3            src/estimation/estimation.py: smoothing and votes ACROSS frames,
   │                                  plus the IMU and encoders -> EstimationPacket (the only input navigation gets)
   └─ navigate()   Navigation         src/navigation/navigation.py: Navigation.update(packet) -> Command
                                      then enforce(): a command that breaks the rules becomes BRAKE
   ▼
Command(left, right, brake)
   │  src/main.py (course run) or src/navigation_linker.py (debug run)
   ▼
MotorController.drive(left, right) / brake()      src/peripherals/drive.py
```

Two things never change:
- **Navigation only sees an `EstimationPacket`** and only returns a `Command`. It never reads a sensor or drives a motor itself. That's why it can be tested on a laptop with made-up packets.
- **The course run and the linkers call the same code.** `src/main.py` drives and records nothing. The linkers drive and record everything. The tests check they give the same commands frame by frame, so a linker's recording is what the course run would have done.

---

## 2. Inside `Navigation.update()`: who decides

`Navigation` (`src/navigation/navigation.py`) is a short list. Every frame:

1. **`StopLineTracker`** (`stop_line.py`) updates first. It follows one stop line: seen → gone from the bottom of the view (an intersection starts, and the route moves to its next step) → **reached** 1.5 s later, when the robot is at the line.
2. Then each **rule** is asked, in priority order. The **first one that returns a `Command` wins**:

| Priority | Rule | File | Speaks when |
| --- | --- | --- | --- |
| 1 | `StopSignRule` | `stop_sign.py` | At the line with a stop sign seen: stop, hold 2 s, go |
| 2 | `TrafficLightRule` | `traffic_light.py` | At the line with the light red: wait |
| 3 | `IntersectionRule` | `intersection.py` | From the line leaving the view until the lane is back: **straight, left or right** |
| 4 | `EndOfCourseRule` | `end_of_course.py` | The lane is lost: creep, then end the run |
| — | `LaneKeepingNavigator` | `lane_keeping.py` | No rule spoke: follow the lane |

Every rule has the same shape:

```python
def update(self, packet, held=False) -> Command | None:
    # None = "nothing to say this frame"
    # held = a higher rule already decided this frame (e.g. stopped at a stop sign):
    #        keep your state up to date, but don't run your timers
```

Every rule also keeps a `record` dict saying why it decided. That dict becomes the `nav.csv` row and the strip under `nav.avi`, which is how you see what the robot was thinking.

---

## 3. Where the turn lives now

Everything the robot does inside an intersection is in **`src/navigation/intersection.py`**, in three stages:

```
stop line leaves the view                    the line (1.5 s later)                         lane back
        │──────────── to_line ────────────────│──────── turn ────────│──────── exit ────────│
        straight, holding heading on the gyro   left/right duties     straight on the new   lane keeping
                                                until the gyro reads  heading               takes over
                                                85° turned
```

- Which way to go comes from **`route.json`** (`{"maneuvers": ["left", "straight", "right"]}`). `RouteProgress` (`route.py`) counts intersections, and the rule reads this one's maneuver.
- A stop sign or red light at the line wins first (it has higher priority). The turn starts when the robot is let go.

**Where your `turning_sequences.py` went:**

| In `turning_sequences.py` | Now |
| --- | --- |
| `drive_left_turn(0.36, 0.63)` | `LEFT_TURN = Command(0.36, 0.63)` in `intersection.py` |
| `drive_right_turn(0.45, 0.0)` | `RIGHT_TURN = Command(0.45, 0.0)` |
| Forward 1.3 s after the stop line, then turn | Stage `to_line`: straight until the line is reached (`STOP_DELAY_MS`, 1.5 s, in `stop_line.py`) |
| Turn for 2.75 s (left) / 1.62 s (right) | Turn until the **gyro** reads 85° (`TURN_TARGET_DEG`). Your times plus half are the backstop (`LEFT_TURN_MAX_MS`, `RIGHT_TURN_MAX_MS`) |
| `is_stop_line_crossing()` | The tracker's `phase == CROSSING`. The tracker owns "where is the line" |
| Which maneuver to do | `route.json` → `RouteProgress` → the rule |
| `CustomSequenceNavigator` wrapping the navigator | Not needed: the turn is part of a rule, so stop signs and red lights still come first |

Why it moved:
- **The wrapper couldn't import:** `DriveState` doesn't exist in `navigation.py`.
- **It started early:** it began at the first sight of a stop line, far up the image, not when the line passed under the robot.
- **It read missing fields:** it looked for `pkt.stop_line_tracker` and `pkt.reached_line`, which the packet doesn't carry.
- **It skipped the safety rules:** while it ran, stop signs and red lights were never checked.
- **Gyro beats timing:** a timed turn changes with battery charge and floor grip; a gyro target lands the same way every time.

---

## 4. Where do I change…?

| I want to change… | File | Name |
| --- | --- | --- |
| How hard / wide a turn is | `src/navigation/intersection.py` | `LEFT_TURN`, `RIGHT_TURN` |
| How far a turn goes | `src/navigation/intersection.py` | `TURN_TARGET_DEG` (85) |
| When the robot counts as "at the line" (turn start, sign stops) | `src/navigation/stop_line.py` | `STOP_DELAY_MS` (1500) |
| When lane keeping takes back over after a crossing | `src/navigation/intersection.py` | `TWO_BOUNDARY_FRAMES`, `ONE_BOUNDARY_FRAMES`, `MAX_CROSS_MS` |
| Which way at each intersection | `route.json` | `maneuvers` |
| Lane-keeping speed and steering | `src/navigation/lane_keeping.py` | `BASE_SPEED`, `KP_*` |
| Stop sign hold | `src/navigation/stop_sign.py` | `STOP_SIGN_HOLD_TIME_MS` |
| Gyro direction | `src/params.py` | `IMU_YAW_SIGN` (+ = turning right) |
| Gyro bias (the main run, every linker, the drive trial) | `src/config.py` | `GYRO_BIAS_DPS` |
| Motor pins | `src/peripherals/drive.py` | `MotorController.__init__` defaults |
| What the camera accepts as a stop line | `src/config.py` | `MEASURED` → `StopLineFilter(max_tilt_deg=15)` |

Every tuned number is a named constant at the top of its file, with a comment saying where it came from. Change it there, never inline.

---

## 5. Testing a change, smallest first

1. **Unit tests** (any computer, seconds): `python3 -m pytest -q src/tests/test_intersection.py`. These feed the rule made-up packets and check every stage. Add a test when you add behavior.
2. **Whole suite:** `python3 -m pytest -q src/tests`. It includes the checks that production and the linkers still agree.
3. **One intersection on the mat:** `python3 -m src.intersection_linker left --camera`. It gives PASS / CHECK with the heading turned (`intersection_linker.md`).
4. **A full course, recorded:** `python3 -m src.navigation_linker --camera --route my_route.json`.
5. **The real thing:** `python3 -m src.main`.

When something looks wrong on the mat, open that run's `nav.csv`. Its columns, read left to right, answer what you need to know:
- `rule`: who decided;
- `stage`: where in the intersection it was;
- `reason`: why;
- `heading_deg`: how far it has turned;
- `cmd_left` / `cmd_right`: what the motors got.
