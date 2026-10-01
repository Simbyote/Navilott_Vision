# Navigation Contract

> What Navigation receives each frame, what it must give back, and how the robot responds, as proven on the robot by `maneuver_linker`.

Navigation doesn't exist yet. This page fixes the handoff so the vision stack and Navigation build against the same thing. Everything in it was measured on the robot in the 2026-09-30 `maneuver_linker` trials, or is enforced by code. Anything not proven is listed under **Open items**, not stated as fact.

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
| `stop_line_detected`, `stop_line_distance_px` | A stop line says an intersection is coming. It stops the robot only with a stop sign or a red light, once the robot reaches it: 1.5 s after it leaves the bottom of the view |
| `lane_mode` | Phase 2's mode this frame; `two_boundary` = both lane boundaries seen |
| `heading_error`, `yaw_rate` | Estimation's convention is **+ = turning right**. See "Yaw sign" below |
| `left_wheel_cps`, `right_wheel_cps` | Counts per second over the frame window (100 Hz readings grouped per frame by `sensing.py`), + = forward. 0.0 means stopped **or** no encoders |
| `wheel_speed` | Always 0.0 for now (`@TODO`: counts per revolution and wheel diameter). Don't use it |
| `stop_sign_detected` | The sign gate is untuned (see `phase3_estimation.md`, "Open items"). Don't rely on it yet |
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

Gyro bias at rest is about −1.0 °/s raw, +1.0 °/s after the flip. It is measured again at every start and shouldn't be trusted as a constant. Lateral acceleration reads about −0.9 m/s² at rest from the mount's tilt, so only changes in it mean anything.

---

## Yaw sign

This robot's IMU is mounted upside down, so its raw gyro Z reads **+ when the robot turns left**. The maneuver trial measures this with its spin pulses. Estimation's convention is + = right, so `src/peripherals/sensing.py`'s `SensorHub` flips the raw value once, per robot, when the sensors are read (`IMU_YAW_SIGN = -1` in `params.py`). `yaw_rate` and `heading_error` in the packet read **+ = turning right**. Gyro bias is in the same flipped frame: about **+1.0 °/s** at rest on this robot.

---

## Who does what

| Vision stack (Phases 1–3) | Navigation |
| --- | --- |
| One packet per frame, stamped at capture | One `Command` per packet |
| Says how much to trust the lane (`lane_status`) | Decides what to do on `hold` and `stale` |
| Votes `drive_state`, stop sign, stop line | Decides at each stop line: stop, wait, cross |
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

1. **`stop_line.py` `StopLineTracker`** advances, once, for every rule: line in view (APPROACH) → gone from the bottom of the view (CROSSING) → **reached** 1.5 s later (`STOP_DELAY_MS`), since the view ends about 10 cm ahead of the robot and braking at the moment it left stopped the robot short. A line lost while still above `NEAR_BOTTOM_ROWS` (25 rows) is flicker, not reached.
2. **`stop_sign.py` `StopSignRule`:** at a reached line with a stop sign seen in the last 5 s, brake until the wheels read stopped (≤ 20 counts/s each, or after 1 s of braking), hold 2 s, go.
3. **`traffic_light.py` `TrafficLightRule`:** at a reached line with the light red, brake until it isn't. Caution drives on.
4. **`intersection.py` `IntersectionRule`:** from the moment the line leaves the view, drive straight on a gyro heading hold, not on lane keeping, which gets pulled left by the crossing street's boundary. The crossing ends once the robot has reached the line and Phase 2 sees both boundaries (`lane_mode`) for 3 frames in a row, or after 4 s of driving. It is straight through only: **turns are TBD** (Ignacio's logic will plug into this rule, which is what keeps lane keeping out of the intersection).
5. **`end_of_course.py` `EndOfCourseRule`:** the course ends at the mat's edge, where the lane lines end. On a `stale` lane it creeps at `SLOW_DUTY` (0.30; half of 0.40 would stall under the 0.25 stall duty), steering by the heading turned. If the lane is still stale after `END_STALE_MS` (1 s) of that, it brakes and sets `finished`. The count pauses while a higher rule holds the frame: no lane boundaries in the middle of an intersection, or stopped at a sign or light, isn't the end. A dedicated final stop line could replace this later.
6. **`lane_keeping.py` `LaneKeepingNavigator`** when no rule speaks.

**`Navigation.finished`** is the run's end: once set, every command is `BRAKE` (no rule is asked again), and whatever runs the loop (`navigation_linker`, the production run) ends on it.

The first rule that speaks wins. Every rule still sees every frame, told whether a higher one already decided (`held`), so its state stays current: the light is watched during a stop sign's hold, and the crossing's clock and boundary count don't run while the robot is braked at the line.

`Navigation.record` names the deciding part (`rule`), the tracker's phase and that part's own reason, for the linkers' logs and video; it's debug output, not part of the contract.

**Lane keeping** (Ignacio, 2026-09-30) steers by `lane_offset_cm` (`lane_offset` until `cm_per_px` is set) on `vision`, and by `heading_error` on `hold` and `stale`. The offset gain grows with the offset (`kp × (1 + 0.05 × |offset cm|)`, his ae34566). Steering is clamped to ±0.40 around a base duty of 0.40, and a slow wheel is lifted to the stall duty. `steer()` is shared with the intersection and end-of-course rules (with their own base speed and clamp). The gains are bench values; the normalized-offset path assumes 30 cm per unit of `lane_offset` (`NORM_TO_CM`), unmeasured.

**Contract checks:** `Navigation` passes every check. Lane keeping on its own doesn't pass the stale one; slowing and ending are the end-of-course rule's job.

`src/scripts/lane_keeping_demo.py` drives the motors from scripted packets through `Navigation`: lane keeping, then a stop-sign intersection (wheels off the ground): `python3 -m src.scripts.lane_keeping_demo`. `navigation_linker` (`guides/navigation_linker.md`) drives the robot with the whole chain and `Navigation`, and is the step's harness before it goes into `pipeline.py`.

---

## Open items

- **Turns at intersections.** TBD (Ignacio): the crossing goes straight only.
- **End of course.** A lost lane ends the run; measure the roll-out past the lane's end on the mat and tune `END_STALE_MS`. A final stop line, if the course gets one, would be exact.
- **Tuning on the mat:** `STOP_DELAY_MS` (1.5 s), `SIGN_MEMORY_MS` (5 s), `STOPPED_CPS` (20), `MAX_CROSS_MS` (4 s). The sign and traffic-light gates are still uncalibrated.

- **Distance.** The maneuver trial measured leg length with cumulative encoder counts. The packet carries only counts per second, so Navigation integrates `cps × dt` itself. Whether the packet should carry cumulative counts is a contract change to agree on.
- **`0.0` pass-throughs** can mean zero or unavailable (`phase3_estimation.md`).
- **Old packets.** What Navigation does with a packet much older than the last isn't decided (time steps are capped at 0.5 s).
- **`caution`** has no defined behavior.
- **`wheel_speed`** in m/s needs counts per wheel revolution and the wheel diameter.
