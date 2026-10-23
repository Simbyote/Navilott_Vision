# Navigation Linker

> Drive the robot with the whole chain: camera → perception → estimation → navigation → motors, recorded and rendered afterwards.

`navigation_linker` is the debug twin of the main pipeline's Navigation step (`Pipeline.navigate()` in `src/pipeline.py`); `test_pipeline` holds the two to the same commands frame by frame. Every frame runs Phase 2 and the traced Phase 3, hands the packet to the navigation subsystem (`Navigation` in `src/navigation/navigation.py`: the stop sign, red light and intersection rules over lane keeping), checks the command against the contract (`enforce()`, as the pipeline does), and drives the motors with it. Nothing is drawn while the robot moves. The frames are saved in the background and `nav.avi` is made after the run, so what you measure is the real control loop.

**Code:** `src/navigation_linker.py` · **Rig and recorder:** `src/linker_io.py` · **Video:** `src/debugger/debug_navigation.py` · **Navigator:** `src/navigation/lane_keeping.py` · **Contract:** `vision_pipeline/navigation_contract.md` · **Tests:** `src/tests/test_navigation_linker.py`, `test_debug_navigation.py`

---

## 1. What it does, and what stops it

Each frame, in order:

1. read the camera and the sensors (IMU and encoders, grouped per frame);
2. Phase 2 and Phase 3 turn the frame into an `EstimationPacket`;
3. the navigator returns a `Command`. If the command breaks the contract (a duty out of range or below the stall duty, or a brake that carries duty), the linker **brakes instead** and logs why;
4. the motors get the command: `brake()` for a brake, otherwise `drive(left, right)`;
5. the frame and a record of every decision go to the recorder.

What `Navigation` does (`navigation_contract.md`, "The navigation subsystem"):

- **Lane keeping:** 0.40 duty, steering against the lane offset, or against the heading while vision is lost.
- **A stop line** says an intersection is coming. From the moment it leaves the bottom of the view, the robot drives **straight on the gyro** until the lane boundaries are back (both for 10 frames, or one for 40), or 49.7 cm past the line (5.5 s without odometry). No new stop line is taken up while it crosses.
- **With 9.0 cm of the line left ahead** (`REACH_CM`, by odometry; without it, as the line leaves the view) the robot is at it:
  - with a **stop sign** seen in the last 5 s, and no more than 2 s before the line came into view, it stops, holds 2 s, and goes;
  - with a **red light**, it waits for the light to change.
- **Otherwise it crosses.** A red light with no stop line doesn't stop it.

**What ends a run:**

| Ends it | Notes |
| --- | --- |
| **The end of the course** | With the route done (or a `stop_line` finish reached), the lane staying lost (creep at 0.30 for 1 s, then brake) or the finish line ends it: `ended by end of course at step N`. The production run ends the same way |
| **Ended early** | The lane stays lost before the route is done: a safety stop, `ended by ended early (...) at step N` |
| `--max-run-s` (default 30 s) | The linker's safety backstop |
| `--camera-control KEY=VALUE` | A camera exposure or white-balance setting for this run, over `params.CAMERA_CONTROLS`; repeatable, e.g. `--camera-control ae-constraint-mode=highlight --camera-control exposure-value=-1` (`phase1_capture.md`) |
| Ctrl-C | Motors stop first, then everything is saved and the video rendered |
| The source ending | Replays only |
| `--limit N` frames | Mostly for replays |
| An error | Motors stop first; the traceback is saved to `error.txt` |

The motors stop first however the run ends.

**When the motors run:** only with `--camera`, and not with `--no-motors`. `--video` and `--frames` replays never drive the motors or open the sensors. They show what the navigator would have commanded on recorded footage.

---

**The route:** the linker reads `route.json` (`config.ROUTE_PATH`; `navigation_contract.md`, "The route") at startup and prints the plan; `--route FILE` uses another. A bad file stops it before anything opens (exit 2). Every stop line passing under the view is the next intersection.

---

## 2. Before the first run

1. `sudo pigpiod` (or `make -f session.mk session TIME="..."`).
2. **Bench check, nothing moves:** `python3 -m src.navigation_linker --camera --no-motors --no-button --max-run-s 10`. Hold the robot over the mat and check that `nav.csv` shows `steer` on a visible lane and that the video's duty bars lean the right way.
3. **Replay check:** run it on a recorded clip: `python3 -m src.navigation_linker --video runs/<some run>.avi`.

---

**The battery** (with `--camera` only): checked at rest before the start, as in a course run (`production_run.md`), except that on the bench (`--no-motors`) a critical pack only warns, since nothing moves. It's watched from the start: a critical pack ends the run (`ended by battery critical`), and `summary.txt` gets a line with the start, lowest and end voltage. The lowest under load, against `stop_line_cm` and where the robot stopped, shows whether a sagging pack is what moves the stop (`STOP_DELAY_MS` is a time).

## 3. Running it on the mat

Place the robot in its lane, pointing along it, with clear space ahead.

```
python3 -m src.navigation_linker --camera                   # start button, 30 s cap
python3 -m src.navigation_linker --camera --max-run-s 10    # a short first run
python3 -m src.navigation_linker --camera --no-button       # 3 s console countdown instead
python3 -m src.navigation_linker --camera --cm-per-px 0.05  # with a measured ground scale
python3 -m src.navigation_linker --camera --route my_route.json   # another route file
```

Keep a hand near it for the first runs. Ctrl-C stops the motors.

---

## 4. Reading the results

Everything goes to `runs/nav_<timestamp>/`.

**`summary.txt`, `[NAVIGATION]`:**

| Line | What it says | Look for |
| --- | --- | --- |
| `ended by` | What ended the run, and the route step it ended on; motors ON or OFF | `run time cap` is normal; `ended early` means the lane was lost before the route was done |
| `run` | Frames, time, FPS, camera and recorder drops | FPS about 20; recorder drops only mean video gaps |
| `driving` | Frames driving, and what the steering came from (`offset_cm`, `offset`, `heading`) | Mostly `offset` or `offset_cm` on a visible lane. Lots of `heading` means vision kept dropping |
| `decided by` | Frames each part decided: `lane_keeping`, `intersection`, `stop_sign`, `traffic_light`, `end_of_course` | `intersection` at every stop line; `stop_sign` / `traffic_light` only where you expect a stop |
| `braked` | Frames braked and why (`stop_sign_stopping`, `stop_sign_hold`, `red_light`, `end_of_course`, `rejected`, `contract`) | Brakes you didn't expect: a false stop sign shows here |
| `steering \|duty\|` | Mean and largest steering | A max stuck at 0.40 means it hit the clamp |
| `command latency` | Frame in → motors, p50 / p95 / max | Well under the 50 ms frame time |
| `contract` | Commands the linker had to brake | Should be 0: anything else is a navigator bug |

**`nav.csv`**, one row per frame:

- **Timings:** `capture_ms`, `phase2_ms`, `phase3_ms`, `nav_ms`, `latency_ms`.
- **The IMU's reads in the frame window:** `imu_reads` (tried: 5 at 100 Hz and 20 FPS; fewer means the sensor hub's ticks slipped), `imu_read_ms` (the slowest, as the hub's thread lived it: I2C plus waiting for the GIL), `imu_failed` (reads that gave nothing). Empty without the IMU or on a replay. `summary.txt` has the run's p50 / p95 / max read time and reads per frame; its battery line adds the ADS1115's read time (`report.json`: `sensors`, `battery.read_ms`).
- **Packet fields the navigator used:** lane status, offset (and cm), heading, light, stop sign, stop line cm, wheel counts per second, and `distance_cm` (odometry: floor cm driven since the start; blank without encoders or `calibration/odometry.json`, `calibrate_odometry.md`).
- **The decision:** `rule` (which part decided), `phase` (the stop-line tracker: idle, approach, crossing), `line_left_cm` (cm of the tracked line still ahead, by odometry from its last sighting; blank without odometry or the stop-line table), `step` and `maneuver` (the route: `2/3 right`), `stage` (inside an intersection: `to_line`, `turn`, `exit`), `stage_cm` (odometry cm driven in that stage; blank without odometry: how `ADVANCE_CM` and `MAX_CROSS_CM` are measured), `turn_end` and `heading_deg` (the intersection's heading turned), `yaw_rate`, `yaw_deg` and `yaw_span_s` (the frame's turn integrated reading by reading, and the seconds it covers: what the heading adds, and what `intersection_linker` replays), `reason` and `source`, `steer`, the command sent, `battery_v` and `battery_state` (the pack's smoothed volts, to the millivolt, and OK / WARNING / CRITICAL; empty without the ADC or on a replay), and `event` (set on a change of reason). `lane_mode` is Phase 2's lane mode, which ends a crossing at `two_boundary`.

**`nav.avi`**: the Phase 3 video with a strip under it:

- **First line:** DRIVE (green) or BRAKE (red) with the reason, and `[rule / phase / step / stage]`.
- **Second line:** the command, the steering and what it came from, the lane and the heading.
- **Duty bars:** one per wheel. The bar fills right of center for forward (green) and left for reverse (red); the amber ticks are the stall duty.
- **Last line:** stop line, light, sign, wheel speeds and latency.

`p3.csv` is the same as `phase3_linker`'s, to trace a bad command back to bad input.

Re-render a run's video later with `python3 -m src.navigation_linker --render runs/nav_<timestamp>`.

---

## 5. Known limits

- **End of course:** a lost lane ends the run after ~1.35 s (Phase 3's hold, then 1 s creeping). Glare or a sharp curve that loses the lane that long ends it too; check `lane_stale_slow` in `nav.csv` and measure how far past the lane's end the robot rolls.
- **Stop line timing:** the robot has to stop as soon as it can at a stop line, so the hanging traffic light stays in view. With odometry (calibrated) it goes by cm: it is at the line when `REACH_CM` (9.0, `src/navigation/stop_line.py`) of it is left ahead, a guess from how far ahead a line still shows before it leaves the view. Turns then drive `ADVANCE_CM` (22.3, `intersection.py`) into the intersection first, set from `stage_cm` where the timed advances ended. Without odometry the same two go by time: `STOP_DELAY_MS` (0: stop as the line leaves the view) and `ADVANCE_MS` (2.5 s). If it stops short or long of the line, tune the first; if turns cut the corner or overshoot, the second. nav.csv's `line_left_cm` (cm of the line still ahead, by the encoders) and `stage_cm` show where each landed.
- **Turns:** tested one intersection at a time with `intersection_linker` (`intersection_linker.md`); tune `TURN_TARGET_DEG` there.
- **Angled approaches:** a robot that comes at a stop line off square carries the angle through the crossing unless `TILT_TO_HEADING` (`intersection.py`) is set; then the crossing's heading starts at the approach angle the line's tilt gives (`approach_deg` in nav.csv), the heading hold squares the robot to the street, and a turn's 85° is measured from the street. See "Measuring `TILT_TO_HEADING`" below.
- **Intersection count:** a missed or false stop line shifts the route; check `step` in `nav.csv`.
- **Stop sign and traffic light:** their gates are uncalibrated, so both can be missed or falsely seen.
- **Gains:** Bench values. His normalized-offset path assumes 30 cm per unit of `lane_offset`, which isn't measured; pass `--cm-per-px` once the ground scale is known and the navigator steers by cm.

### Measuring `TILT_TO_HEADING`

The scale from the stop line's tilt in the image to the robot's angle on the floor. From the low camera a line looks flatter than it is, so it's larger than 1; it's negative because a robot turned right sees the line's left end nearer.

1. Park the robot on the mat facing a stop line square, its front edge ~15 cm from the line (the line in the lower half of the view). Run `python3 -m src.phase3_linker --camera --limit 100` and note the median `stop_line_tilt_deg` in `p3.csv`. That's `TILT_ZERO_DEG`: near 0 unless the camera sits a little rotated.
2. Turn the robot about its center to a known angle, **right** of square, e.g. 8°: a protractor, or a strip of tape at that angle to line it up against. Same run, same column.
3. `TILT_TO_HEADING = angle / (tilt at the angle − tilt square)`, e.g. 8 / (−2.6 − 0.1) ≈ −2.96. Check it on the left: −8° should read about the same tilt with the sign flipped.
4. Keep the angle under what the geometry's tilt gate passes (`StopLineFilter.max_tilt_deg` in `config.py`): past it the line isn't detected at all, which is also the largest approach this can correct (gate × the scale).

The `detect-range` routine does the same with prompts and fits all the angles at once: `make routine-detect-range ARGS="--set target=line --set gaps=15 --set angles=-8,-4,0,4,8"` prints `TILT_ZERO_DEG` and `TILT_TO_HEADING` (a least-squares fit of tilt against angle), and its rows show at which angle the tilt gate starts refusing the line.

With it set, `intersection_linker` crossings from a deliberately angled start should end with `heading_deg` near the street's (`approach_deg` in nav.csv says what the line read).

| Symptom | Likely cause |
| --- | --- |
| Robot steers the wrong way | Check the lane offset sign in `p3.csv` against the video before touching the navigator |
| `braked ... contract N` | A navigator bug; `nav.csv`'s `event` column says which rule |
| FPS well under 20 | Check `phase2_ms` in `nav.csv`; `--no-display` so the render doesn't run afterwards |
| Robot never moves | `motors OFF` in the summary (replay or `--no-motors`), or it braked from the first frame: check `reason` |
