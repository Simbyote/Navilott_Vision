# Navigation Contract

> What Navigation receives each frame, what it must give back, and how the robot responds, as proven on the robot by `maneuver_linker`.

This page fixed the handoff before Navigation existed (2026-09-30), so the vision stack and Navigation could build against the same thing. Navigation is now built against it (`src/navigation/`, "The navigation subsystem" below). Everything in it was measured on the robot in the 2026-09-30 `maneuver_linker` trials, or is enforced by code. Anything not proven is listed under **Open items**, not stated as fact.

**Code:** `src/navigation/navigation_contract.py` (`Command`, `BRAKE`, `Navigator`, `command_problems`; re-exported by `src/navigation/navigation.py`) · **Packet:** `src/estimation/estimation.py` (`EstimationPacket`) · **Checks:** `src/tests/navigation_checks.py` · **Tests:** `src/tests/test_navigation_contract.py`

---

## The interface

```python
class Navigator(Protocol):
    def update(self, packet: EstimationPacket) -> Command: ...
    def reset(self) -> None: ...
```

- `update()` is called **once per camera frame, in frame order**, with that frame's packet. It returns a `Command` **every time**: it never blocks or sleeps, and it never raises on ordinary input (a stale lane, no detections, `None` distances).
- `reset()` forgets all state, as at the start of a run.
- The pipeline drives whatever comes back: `motor.brake()` if `cmd.brake`, otherwise `motor.drive(cmd.left, cmd.right)`.

| `Command` field | Meaning |
| --- | --- |
| `left`, `right` | Duty per wheel for `MotorController.drive()`, in [−1, +1], **+ = forward** |
| `brake` | Short-brake both motors (`MotorController.brake()`); `left` and `right` must then be 0 |

`BRAKE = Command(brake=True)` is "stop and hold still". `Command()` (zero duty, no brake) **coasts**, which is different; see below.

### Command rules (`command_problems()`)

1. Each duty is within [−1, +1].
2. A wheel that is driven gets at least `STALL_DUTY` (0.25) in magnitude. Below that the N20s stall under load. 0 means "that wheel doesn't drive".
3. A brake carries zero duty.

---

## The packet

`EstimationPacket` is what `maneuver_linker` produces through `run_phase3_chain()`, plus `lane_mode` (added 2026-10-01: this frame's Phase 2 lane mode, passed through unfiltered, so Navigation can tell when both lane boundaries are back after an intersection). The field table is in `phase3_estimation.md`, "Contract". The rules Navigation must follow:

| Field | Rule |
| --- | --- |
| `lane_offset` | + = robot **right** of lane center, so correcting it means steering **left** (left duty < right duty) |
| `lane_status` | Steer by `lane_offset` only on `vision` or `hold`. On `stale`, the offset is an old value: **creep at slow speed, then stop**. A lane that stays lost is the end of the course (decided 2026-10-01) |
| `drive_state` | `stop` (red) **stops the robot at a stop line**, not before it and not without one; `go` and `caution` drive on (decided 2026-10-01) |
| `stop_line_detected`, `stop_line_distance_px` | A stop line says an intersection is coming. It stops the robot only with a stop sign or a red light, once the robot reaches it: with `REACH_CM` (9.0 cm) of it left ahead by odometry, or as it leaves the bottom of the view without |
| `lane_mode` | Phase 2's mode this frame; `two_boundary` = both lane boundaries seen |
| `obstacle_detected`, `obstacle_distance_cm` / `_px`, `obstacle_clipped` | Something in the path within reach (12 cm from the front edge, or 24 rows with no cm, or clipped) **brakes the robot, and it stays braked until nothing has been seen for 1 s**; then it drives on (decided 2026-10-12). Ahead of every other rule. The lane it hides doesn't count toward the end of the course |
| `heading_error`, `yaw_rate` | Estimation's convention is **+ = turning right**. See "Yaw sign" below |
| `left_wheel_cps`, `right_wheel_cps` | Counts per second over the frame window (100 Hz readings grouped per frame by `sensing.py`), + = forward. 0.0 means stopped **or** no encoders. The packet's only wheel speed: there is no m/s field (see "Open questions") |
| `distance_cm`, `left_distance_cm`, `right_distance_cm` | Odometry: floor cm driven since the run's first encoder reading, the axle's midpoint and each wheel, + = forward. `None` without encoders or without `calibration/odometry.json`, so a rule that drives by distance needs a fallback (time) for that case. The stop at the line (`REACH_CM`), the intersection's advance (`ADVANCE_CM`) and the exit's limit (`MAX_CROSS_CM`) go by it; without odometry each falls back to time. The intersection's records carry `stage_cm`, the cm driven in the stage, which nav.csv logs |
| `stop_sign_detected` | Phase 3's vote gate (0.45) isn't measured on course runs yet (see `phase3_estimation.md`, "Open items") |
| `timestamp_ms` | Capture time on `time.monotonic_ns() // 1_000_000`. Differences between packets give dt |

---

## Timing

| Quantity | Measured | Source |
| --- | --- | --- |
| Frame rate | ~19.8 FPS | `maneuver_linker`, 2026-09-30 |
| Frame dt | median 0.050 s, max 0.159 s | same |

`update()` is called at the frame rate, so a command stands for about 50 ms, occasionally up to ~160 ms. Navigation should compute its time steps from `timestamp_ms` rather than assume 50 ms.

---

## How the robot responds

Measured on this robot with `maneuver_linker`, 2026-09-30:

| Command | Response |
| --- | --- |
| Forward, 0.40 on both wheels | ~1166 counts/s per wheel |
| Spin left in place | `(−s, +s)`; right is `(+s, −s)` |
| Spin at 0.45 | ~96 °/s, ~1350 counts/s per wheel |
| Zero duty (coast) | Keeps turning ~0.15–0.18 s; the 180° turn ended at 186–188° |
| `BRAKE` | Stops sharply. With it, a gyro target of **171°** gives a real **180°** turn |
| Below 0.25 duty | Stalls under load |

Gyro bias at rest is about −1.0 °/s raw in these trials (`config.GYRO_BIAS_DPS` holds −0.90, from the 2026-09-29 `phase3_linker` run, found a bit better than the trials' −1.0); with `IMU_YAW_SIGN` 1, the hub's frame is the raw one. It is measured again at every start and shouldn't be trusted as a constant. Lateral acceleration reads about −0.9 m/s² at rest from the mount's tilt, so only changes in it mean anything.

---

## Yaw sign

This robot's IMU is mounted upside down, so its raw gyro Z already reads **+ when the robot turns right**, Estimation's convention (checked on the robot 2026-10-01, after the motor sides were fixed; the −1 measured on 2026-09-30 was with them swapped). The maneuver trial measures this with its spin pulses. `src/peripherals/sensing.py`'s `SensorHub` multiplies the raw value by `IMU_YAW_SIGN` once, per robot, when the sensors are read: 1 on this robot (`params.py`). `yaw_rate` and `heading_error` in the packet read **+ = turning right**. Gyro bias is in the same frame: **−0.90 °/s** at rest on this robot (`config.GYRO_BIAS_DPS`).

---

## Who does what

| Vision stack (Phases 1–3) | Navigation |
| --- | --- |
| One packet per frame, stamped at capture | One `Command` per packet |
| Says how much to trust the lane (`lane_status`) | Decides what to do on `hold` and `stale` |
| Votes `drive_state`, stop sign, stop line, obstacle | Decides at each stop line: stop, wait, cross; and for an obstacle: brake, wait, go |
| Passes IMU and encoder readings through | Uses them: speed control, turns, distance |
| Never drives the motors | Never reads the camera or the sensors directly |

---

## Checking a Navigator

`src/tests/navigation_checks.py` holds the checks every Navigator must pass, so any implementation can be tested against the contract without the robot:

| Check | What it holds |
| --- | --- |
| `check_commands(nav, packets)` | Every packet gets a `Command` that passes `command_problems()` |
| `check_stale_lane_slows_then_stops(nav)` | A `stale` lane: mean duty at most 0.30 (+0.05 for a lifted slow wheel), stopped within 40 frames (~2 s), and staying stopped |
| `check_stops_at_a_red_line(nav)` | A red light at a stop line: no braking while the line is in view, then braking after it, for as long as it's red |
| `check_crosses_a_green_line(nav)` | A stop line with a green light and no sign: never brakes, keeps driving |
| `check_ignores_a_red_light_without_a_line(nav)` | A red light with no stop line doesn't stop the robot |
| `check_stops_at_a_stop_sign_line_then_goes(nav)` | A stop sign at a stop line: no braking while the line is in view, a stop after it, then driving on |
| `check_goes_when_the_light_turns_green(nav)` | Waiting at a red line, green lets the robot drive on |
| `check_steers_toward_center(nav)` | On `vision`, driving forward with the robot right of center, left duty < right duty; and the mirror case |

`contract_problems(nav)` runs them all. Each returns a list of problems (`[]` when the navigator passes), so a test can print them all. `test_navigation_contract.py` checks the checks themselves with a navigator that obeys the contract (`Navigation`) and ones that each break one rule.

### Keeping hardware out

`navigation_contract.py` holds only the contract: no pigpio, no motor code, no copies of the packet. The navigation subsystem is pure logic too: it imports `EstimationPacket` from `estimation.py` and `Command` from the contract, and returns commands. Only whatever runs the loop (a script, a linker, the pipeline) opens the motors, through `MotorController` in `peripherals/drive.py`, importing it inside the function that opens hardware. `test_navigation_contract` loads the contract, the orchestrator and every rule with pigpio blocked, as on a laptop, to keep it that way.

---

## The navigation subsystem

`src/navigation/navigation.py`'s `Navigation` is the Navigator the robot drives with. It orchestrates the subsystem, and each decision lives in its own file as a **rule** that returns a `Command` or `None` ("nothing to say"). Every frame:

1. **`stop_line.py` `StopLineTracker`** advances, once, for every rule: line in view (APPROACH) → its near edge at the bottom of the view, the voted line reading clipped (CROSSING; `entered` is True on that one frame; `CROSS_ON_CLIP`, since 2026-10-20: waiting for the whole tape to vanish braked a tape's width late, ten stop-distance stops 1.0 cm from the line) → **reached** on the next frame (`STOP_DELAY_MS` 0, tuned on the course: the traffic light hangs over the intersection, and any drive past the line carries the robot under it and out of view; braking as the line leaves keeps the whole light in one frame. 1.5 s stopped it past the line and 0.5 s at it, 2026-10-05). `seen_ms` names when the line first came into view, for the stop sign rule. A line lost while still above `NEAR_BOTTOM_ROWS` (25 rows) is flicker, not reached. Each `entered` is one intersection: the route's progress advances on it (below). While an intersection is being crossed, `Navigation` keeps the tracker from taking up a new line (`accept_new`), so a line-like mark on the far side can't restart the crossing or count as another intersection (2026-10-01 run).
   By distance instead, when `REACH_CM` is set and the packet has odometry and a cm for the line: each fresh sighting puts the line at an odometry position (`distance_cm` + `stop_line_distance_cm`; a value Phase 3 holds through a missed frame isn't fresh, nor is a clipped one, which reads 0 cm though the line is still at the view's edge), and once it has left the view the line is reached on the frame that leaves `left_cm` nearest `REACH_CM` (less than half the last frame's travel to go). Where in a frame the line left the view stops mattering. `REACH_CM` is 9.0 cm, a guess from how far ahead a stop line still shows before it leaves the view; set to `None` (or without odometry) the stop goes by `STOP_DELAY_MS`. nav.csv's `line_left_cm` logs `left_cm` either way, so the stops show what to set. The earliest a line can be reached is still the frame after it leaves the view.
2. **`obstacle.py` `ObstacleRule`:** the first rule, above the stop sign: nothing else the robot does is worth hitting something for. When Phase 3's obstacle vote has something in the path with its base within `OBSTACLE_STOP_CM` (12 cm; with no cm, `OBSTACLE_STOP_PX` rows of the view bottom), or already out of view (clipped), it brakes. It keeps braking until no obstacle has been seen at all, at any distance, for `OBSTACLE_CLEAR_MS` (1 s): an object the robot stopped just short of reads a little beyond the stop distance, and releasing on "not near" would creep the robot into it a stop at a time. The distance comes from the stop-line table, measured from the robot's front edge, and the camera's view starts ~10.2 cm ahead of that edge, so "within 10 cm" can't be seen as a distance. The stop distance sits just beyond the view's edge instead: the brake starts while the base is still in view, and the coast past a zero duty (~0.15–0.18 s, 2026-09-30) plus a frame or two of voting ends the robot near 10 cm. While it brakes the rules below are held: the end of course doesn't count a lane hidden behind the obstacle, and a crossing's timers wait.
3. **`stop_sign.py` `StopSignRule`:** at a reached line with a stop sign seen in the last 5 s (`SIGN_MEMORY_MS`) and no more than 2 s before that line came into view (`SIGN_LEAD_MS`: a ghost sign seen earlier on the way, a leg or a chair read as a sign, can't stop the robot at an unrelated line), brake until the wheels read stopped (≤ 20 counts/s each, or after 1 s of braking), hold 2 s, go.
4. **`traffic_light.py` `TrafficLightRule`:** at a reached line with the light red or yellow in the last 0.5 s (`RED_MEMORY_MS`), brake until it has read green or nothing for 0.25 s (`RELEASE_MS`). Yellow stops too (2026-10-07): the overexposed red LED reads yellow from some angles, and a real yellow turns red anyway. While it brakes, the intersection rule is held (its timers don't run), and the route's maneuver runs once it lets go. The memory and the release time keep one dropped frame from running the light or letting the robot go on a red (2026-10-06).
5. **`intersection.py` `IntersectionRule`:** everything the robot does inside an intersection, in four stages, from the moment the line leaves the view (lane keeping would be pulled off by the crossing street's lines):
   - **`to_line`:** straight to the line on a gyro heading hold.
   - **`advance`** (the route's `left` / `right` only), from the line: on into the intersection on the heading hold for `ADVANCE_MS` (2.5 s, tuned on the course), since the stop lines sit back in the street and the robot stops as the line leaves the view. With odometry (`distance_cm`) and `ADVANCE_CM` set, it goes that many cm instead: a time moves with the speed, and the heading hold's speed with the battery.
   - **The approach angle:** the crossing's heading starts at the angle the robot came at the line, not 0: the tracker's median of the line's last 5 fresh tilts in the image (`stop_line_tilt_deg`, `approach_tilt_deg`), less `TILT_ZERO_DEG`, × `TILT_TO_HEADING`, capped at 10° (`MAX_APPROACH_DEG`). The heading hold then squares the robot to the street, and the turn's 85° is from the street. `TILT_TO_HEADING` is −4.3 and `TILT_ZERO_DEG` 0.19 ("Measuring `TILT_TO_HEADING`" below); `None` starts every crossing at 0. nav.csv logs `approach_deg`.
   - **`turn`** (the route's `left` / `right` only), after the advance: wheel speeds, tuned on the mat as duties (left `(0.48, 0.73)`, right `(0.78, 0.25)`, 2026-10-19) and driven as encoder counts/s (duty × `CPS_PER_DUTY`, 3000) by `wheel_speed.WheelSpeed`, so a draining pack doesn't shrink the arc: a feedforward at each wheel's counts/s-per-duty gain, learned from steady lane-keeping frames, plus a PI trim on the measured speed. Without encoders (every reading 0.0) it drives the tuned duties as before. nav.csv logs the targets and gains (`target_*_cps`, `gain_*`). The turn runs until the gyro reads 85° turned that way (`TURN_TARGET_DEG`), or the turn's time limit if it never does (`LEFT_TURN_MAX_MS` 4.1 s, `RIGHT_TURN_MAX_MS` 2.4 s: the open-loop turn times plus half). A stop sign or red light at the line holds the robot first; held frames don't use the turn's time.
   - **`exit`:** straight on the heading the turn ended on, until both boundaries are back for 10 frames (`TWO_BOUNDARY_FRAMES`), at least one for 40 (`ONE_BOUNDARY_FRAMES`), or 5.5 s of driving (`MAX_CROSS_MS`; 49.7 cm of odometry instead, `MAX_CROSS_CM`, when the packet has odometry), tuned on the mat 2026-10-19/20. Fewer frames ended the crossing as soon as the far side's lanes showed, and the next lane's stop line was taken up right after. Then lane keeping takes over.

   Straight skips the turn. Its record carries the route step and maneuver, the `stage`, the heading turned, and `turn_end` (`gyro target` or `time limit`).
   - **The heading** adds each packet's `yaw_deg` (the sensor window integrated reading by reading) less the gyro bias over `yaw_span_s`, so a long frame gap during a turn is counted in full. A packet without it (a replay logged before it existed) falls back to `yaw_rate` times the frame step, capped at `MAX_DT_MS`.
6. **`end_of_course.py` `EndOfCourseRule`:** ends the run, how depending on the route (below). On a `stale` lane it creeps at `SLOW_DUTY` (0.30; half of 0.40 would stall under the 0.25 stall duty), steering by the heading turned. If the lane is still stale after `END_STALE_MS` (1 s) of that, it brakes and sets `finished`: `outcome` is `finished` when the route is done and finishes at the edge, otherwise `ended_early` (a safety stop off course), with `end_step` the step it got to. With a `stop_line` finish, it brakes at the finish line once the tracker reaches it (`finished`). The count pauses while a higher rule holds the frame: no lane boundaries in the middle of an intersection, or stopped at a sign or light, isn't the end.
7. **`lane_keeping.py` `LaneKeepingNavigator`** when no rule speaks.

**`Navigation.finished`** is the run's end: once set, every command is `BRAKE` (no rule is asked again), and whatever runs the loop (`navigation_linker`, the production run) ends on it. `Navigation.outcome` (`finished` / `ended_early`) and `end_step` say how.

### The route (`route.py`)

The course plan: one maneuver per intersection, read from `config.ROUTE_PATH` (`vision_stack/route.json`) and checked once at startup, so it can be edited between runs without touching code:

```json
{"maneuvers": ["left", "straight", "right"], "finish": "edge"}
```

- **`maneuvers`:** `straight`, `left` or `right`, in order; case-insensitive. The intersection rule drives each (item 5 above).
- **`finish`** (default `edge`):
  - `edge`: after the last maneuver, a lane that stays lost (the mat's edge) is the finish. Extra intersections past the plan are crossed straight.
  - `stop_line`: the first stop line after the last maneuver is the finish line; the robot stops at it once the tracker reaches it, like any stop.
- **Ended early:** a lane that stays lost before the route is done is a safety stop, `outcome` `ended_early`, with the step reached.
- A bad file (missing, bad JSON, unknown key or name) is a `RouteError` at startup, before anything opens.

`RouteProgress` (`Navigation.progress`) counts intersections: every stop line `entered` is the next step. `Navigation.record["step"]` labels it: `0/3` before the first, `2/3 right`, `4/3 finish`, `5/3 extra`. `Route.describe()` gives the startup screen's lines.

### Wheel speed (`wheel_speed.py`)

A duty is a share of the battery's voltage, so the same duty turns the N20s slower as the pack drains (12.6 V full, 9.9 V critical): a turn tuned as duties on a fresh pack draws a smaller, slower arc on a tired one, and its inner wheel stalls sooner. The encoders measure what the wheels actually do, so `WheelSpeed` asks for a speed in counts/s and finds the duty:

```
duty = target / gain  +  kp × error  +  ∫ ki × error
```

`gain` (counts/s per unit duty) is the feedforward, and it's learned while the robot drives: on every steady lane-keeping frame the measured speed over the commanded duty moves it by `LEARN_ALPHA`, within `GAIN_LIMITS`, so by the time a turn starts it already reflects the day's voltage and floor. The PI trim takes out what's left; its integral runs only once the error is within `INTEGRATE_BAND` of the target, so it doesn't wind up while the wheel spins up. A packet's measured speed covers the frame window that ends with it, so it answers the command sent the frame before, and `observe()` is given that command. Without encoders (every reading 0.0 while driving) it's the feedforward alone at the configured gain: the tuned duties, unchanged.

The first rule that speaks wins. Every rule still sees every frame, told whether a higher one already decided (`held`), so its state stays current: the light is watched during a stop sign's hold, and the crossing's clock and boundary count don't run while the robot is braked at the line.

`Navigation.record` names the deciding part (`rule`), the tracker's phase and that part's own reason, for the linkers' logs and video; it's debug output, not part of the contract.

**Lane keeping** (2026-09-30) steers by `lane_offset_cm` (`lane_offset` until `cm_per_px` is set) on `vision`, and by `heading_error` on `hold` and `stale`. The offset gain grows with the offset (`kp × (1 + 0.05 × |offset cm|)`, commit ae34566). Steering is clamped to ±0.40 around a base duty of 0.40, and a slow wheel is lifted to the stall duty. `steer()` is shared with the intersection and end-of-course rules (with their own base speed and clamp). The gains are bench values; the normalized-offset path assumes 30 cm per unit of `lane_offset` (`NORM_TO_CM`), unmeasured.

**Contract checks:** `Navigation` passes every check. Lane keeping on its own doesn't pass the stale one; slowing and ending are the end-of-course rule's job.

`navigation_linker` (`make navigate`, or `make nav-dry` with the motors off) drives the robot with the whole chain and `Navigation`, recording `nav.csv` and the frames for `make render` and `make nav-report`; it is the debug twin of the Navigation step. `intersection_linker` (`make intersection MANEUVER=left`) runs one intersection per run and judges it PASS or CHECK.

**In the pipeline:** `src/pipeline.py` runs Navigation as its last stage, `Pipeline.navigate(packet)`: `Navigation.update()`, then `enforce()` (`navigation_contract.py`), which returns `BRAKE` (and the reasons, on `Pipeline.last_problems`) for a command that breaks the contract. `navigation_linker` drives through the same `enforce()`. `Pipeline.step()` is frame in, `Command` out; driving the motors with it is the run loop's job: `src/main.py`, the production run (`make run`). The route comes in as `Pipeline(route=...)`, and the intersection's gyro bias is Phase 3's (`Phase3Config.gyro_bias_dps`). `test_pipeline` holds the pipeline's commands to `navigation_linker`'s motor calls frame by frame over `course_sequence()`, a synthetic course in which every rule decides some frame.

### Measuring `TILT_TO_HEADING`

The scale from the stop line's tilt in the image to the robot's angle on the floor. From the low camera a line looks flatter than it is, so the scale is larger than 1; it's negative because a robot turned right sees the line's left end nearer.

`make routine-detect-range ARGS="--set target=line --set gaps=15 --set angles=-8,-4,0,4,8"` parks the robot at each angle, fits tilt against angle by least squares and prints both values. By hand: park square to a stop line ~15 cm out and note the median `stop_line_tilt_deg` in `p3.csv` (`python3 -m src.phase3_linker --camera --limit 100`), which is `TILT_ZERO_DEG`; turn the robot about its center to a known angle right of square and read it again; `TILT_TO_HEADING = angle / (tilt at the angle − tilt square)`. Keep the angles inside the geometry's tilt gate (`StopLineFilter.max_tilt_deg`, 5° in `MEASURED`): past it the line isn't detected at all, which is also the largest approach this can correct.

The values in `intersection.py` come from that routine on 2026-10-20: `target=line`, 10 cm, −6 to 6°. The tilts read +1.39, +1.26, +0.03, −0.49 and −1.22, and redone looks agreed within 0.1–0.3°. Check the sign on the robot: wrong, it doubles the approach error instead of removing it. With it set, `intersection_linker` crossings from a deliberately angled start should end with `heading_deg` near the street's.

### Reading a run

| Symptom | Likely cause |
| --- | --- |
| Robot steers the wrong way | Check the lane offset's sign in `p3.csv` against the video before touching the navigator |
| The summary's `contract  N commands rejected and braked` | A navigator bug; `nav.csv`'s `rule` and `event` columns say which rule |
| Stops short of or past the line | `REACH_CM` (or `STOP_DELAY_MS` without odometry); `line_left_cm` shows where each stop landed |
| Turns cut the corner or overshoot | `ADVANCE_CM` (`ADVANCE_MS`), then `TURN_TARGET_DEG`; `stage_cm` shows where each advance ended |
| A maneuver at the wrong intersection | A missed or false stop line shifted the route; check `step` in `nav.csv` |
| Run ends on a straight | The lane was lost for ~1.35 s (Phase 3's hold, then `END_STALE_MS` creeping): glare or a sharp curve; check `lane_stale_slow` |
| FPS well under 20 | `phase2_ms` in `nav.csv`; run with `--no-display` |
| Robot never moves | `motors OFF (dry run)` in the summary (a replay or `--no-motors`), or it braked from the first frame: check `reason` |

---

## Open items

- **Turns at intersections.** Built (2026-10-02); tune `TURN_TARGET_DEG` and the duties on the mat with `intersection_linker`.
- **End of course.** A lost lane ends the run; measure the roll-out past the lane's end on the mat and tune `END_STALE_MS`. A final stop line, if the course gets one, is the route's `stop_line` finish.
- **Counting intersections.** Every stop line passing under the view counts as one. A missed line (glare) or a false one shifts every later maneuver; watch the `step` column in `nav.csv`.
- **Tuning on the mat:** `REACH_CM` (9.0 cm, a guess from how far ahead a stop line still shows before it leaves the view: the robot stops as soon as it can so the hanging traffic light stays in view; `STOP_DELAY_MS`, 0, is the fallback without odometry), `ADVANCE_CM` (22.3 cm) and `MAX_CROSS_CM` (49.7 cm), both from the odometry's `stage_cm` where the timed advances and exits ended (2026-10-20; `ADVANCE_MS`, 2.5 s, and `MAX_CROSS_MS`, 5.5 s, without odometry), `SIGN_MEMORY_MS` (5 s), `SIGN_LEAD_MS` (2 s), `RED_MEMORY_MS` (0.5 s), `RELEASE_MS` (0.25 s), `STOPPED_CPS` (20), `TWO_BOUNDARY_FRAMES` (10), `ONE_BOUNDARY_FRAMES` (40). The traffic light runs glow mode, tuned on the course's LEDs (2026-10-06, `phase2_perception.md`). The stop sign's shape gates were fit to the course's sign on run frames (2026-10-20, `config.py`); its range and Phase 3's 0.45 vote gate aren't measured yet (`requirements.md`, D2).

- **`0.0` pass-throughs** can mean zero or unavailable (`phase3_estimation.md`).
- **Old packets.** What Navigation does with a packet much older than the last isn't decided (time steps are capped at 0.5 s).
- **`caution`** (yellow) stops the robot at the line like red, since 2026-10-07 (`traffic_light.py`): a red LED's ring can read orange, inside yellow's band, and a red misread as yellow drove through.
- **Wheel speed in m/s.** The packet's `wheel_speed` was removed (2026-10-02): nothing set it, so it was always 0.0. Speed is `left_wheel_cps` / `right_wheel_cps`; × `Phase3Config.cm_per_count` (`calibration/odometry.json`) gives cm/s.
