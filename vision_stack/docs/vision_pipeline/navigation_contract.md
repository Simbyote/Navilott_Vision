# Navigation Contract

> What Navigation receives each frame, what it must give back, and how the robot responds, as proven on the robot by `maneuver_linker`.

Navigation doesn't exist yet. This page fixes the handoff so the vision stack and Navigation build against the same thing. Everything in it was measured on the robot in the 2026-09-30 `maneuver_linker` trials, or is enforced by code. Anything not proven is listed under **Open items**, not stated as fact.

**Code:** `src/navigation.py` (`Command`, `BRAKE`, `Navigator`, `command_problems`) · **Packet:** `src/estimation.py` (`EstimationPacket`) · **Checks:** `src/tests/navigation_checks.py` · **Tests:** `src/tests/test_navigation.py`

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

`EstimationPacket` is unchanged from what `maneuver_linker` produces through `run_phase3_chain()`. The field table is in `phase3_estimation.md`, "Contract". The rules Navigation must follow:

| Field | Rule |
| --- | --- |
| `lane_offset` | + = robot **right** of lane center, so correcting it means steering **left** (left duty < right duty) |
| `lane_status` | Steer by `lane_offset` only on `vision` or `hold`. On `stale`, the offset is an old value: **don't drive forward on it** |
| `drive_state` | On `stop`, **don't drive forward**. `caution` is Navigation's call |
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

This robot's IMU is mounted upside down, so its raw gyro Z reads **+ when the robot turns left**. The maneuver trial measures this with its spin pulses. Estimation's convention is + = right, so `src/sensing.py`'s `SensorHub` flips the raw value once, per robot, when the sensors are read (`IMU_YAW_SIGN = -1` in `params.py`). `yaw_rate` and `heading_error` in the packet read **+ = turning right**. Gyro bias is in the same flipped frame: about **+1.0 °/s** at rest on this robot.

---

## Who does what

| Vision stack (Phases 1–3) | Navigation |
| --- | --- |
| One packet per frame, stamped at capture | One `Command` per packet |
| Says how much to trust the lane (`lane_status`) | Decides what to do on `hold` and `stale` |
| Votes `drive_state`, stop sign, stop line | Stops, waits and goes |
| Passes IMU and encoder readings through | Uses them: speed control, turns, distance |
| Never drives the motors | Never reads the camera or the sensors directly |

---

## Checking a Navigator

`src/tests/navigation_checks.py` holds the checks every Navigator must pass, so any implementation can be tested against the contract without the robot:

| Check | What it holds |
| --- | --- |
| `check_commands(nav, packets)` | Every packet gets a `Command` that passes `command_problems()` |
| `check_no_forward_on_stale(nav)` | A `stale` lane never gets forward drive |
| `check_no_forward_on_stop(nav)` | `drive_state == "stop"` never gets forward drive |
| `check_steers_toward_center(nav)` | On `vision`, driving forward with the robot right of center, left duty < right duty; and the mirror case |

`contract_problems(nav)` runs them all. Each returns a list of problems (`[]` when the navigator passes), so a test can print them all. `test_navigation.py` checks the checks themselves with a navigator that obeys the contract and ones that each break one rule.

---

## Open items

- **Distance.** The maneuver trial measured leg length with cumulative encoder counts. The packet carries only counts per second, so Navigation integrates `cps × dt` itself. Whether the packet should carry cumulative counts is a contract change to agree on.
- **`0.0` pass-throughs** can mean zero or unavailable (`phase3_estimation.md`).
- **Stale and old packets.** What Navigation does after a long `stale` run, or with a packet much older than the last, isn't decided.
- **`caution`** has no defined behavior.
- **`wheel_speed`** in m/s needs counts per wheel revolution and the wheel diameter.
