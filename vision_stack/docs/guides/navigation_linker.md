# Navigation Linker

> Drive the robot with the whole chain: camera → perception → estimation → navigation → motors, recorded and rendered afterwards.

`navigation_linker` is the integration test of the Navigation step before it goes into `pipeline.py`. Every frame runs Phase 2 and the traced Phase 3, hands the packet to the navigation subsystem (`Navigation` in `src/navigation/navigation.py`: the stop sign, red light and intersection rules over lane keeping), checks the command against the contract, and drives the motors with it. Nothing is drawn while the robot moves. The frames are saved in the background and `nav.avi` is made after the run, so what you measure is the real control loop.

**Code:** `src/navigation_linker.py` · **Video:** `src/debugger/debug_navigation.py` · **Navigator:** `src/navigation/lane_keeping.py` · **Contract:** `vision_stack/navigation_contract.md` · **Tests:** `src/tests/test_navigation_linker.py`, `test_debug_navigation.py`

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
- **A stop line** says an intersection is coming. From the moment it leaves the bottom of the view, the robot drives **straight on the gyro** until both lane boundaries are back.
- **1.5 s after the line leaves the view** the robot is at it:
  - with a **stop sign** seen in the last 5 s, it stops, holds 2 s, and goes;
  - with a **red light**, it waits for the light to change.
- **Otherwise it crosses.** A red light with no stop line doesn't stop it.

**What ends a run:**

| Ends it | Notes |
| --- | --- |
| **The end of the course** | With the route done (or a `stop_line` finish reached), the lane staying lost (creep at 0.30 for 1 s, then brake) or the finish line ends it: `ended by end of course at step N`. The production run ends the same way |
| **Ended early** | The lane stays lost before the route is done: a safety stop, `ended by ended early (...) at step N` |
| `--max-run-s` (default 30 s) | The linker's safety backstop |
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
- **Packet fields the navigator used:** lane status, offset (and cm), heading, light, stop sign, stop line cm, wheel counts per second.
- **The decision:** `rule` (which part decided), `phase` (the stop-line tracker: idle, approach, crossing), `step` and `maneuver` (the route: `2/3 right`), `reason` and `source`, `steer`, the command sent, and `event` (set on a change of reason). `lane_mode` is Phase 2's lane mode, which ends a crossing at `two_boundary`.

**`nav.avi`**: the Phase 3 video with a strip under it:
- **First line:** DRIVE (green) or BRAKE (red) with the reason, and `[rule / phase / step]`, with `TBD` on a turn driven straight.
- **Second line:** the command, the steering and what it came from, the lane and the heading.
- **Duty bars:** one per wheel. The bar fills right of center for forward (green) and left for reverse (red); the amber ticks are the stall duty.
- **Last line:** stop line, light, sign, wheel speeds and latency.

`p3.csv` is the same as `phase3_linker`'s, to trace a bad command back to bad input.

Re-render a run's video later with `python3 -m src.navigation_linker --render runs/nav_<timestamp>`.

---

## 5. Known limits

- **End of course:** a lost lane ends the run after ~1.35 s (Phase 3's hold, then 1 s creeping). Glare or a sharp curve that loses the lane that long ends it too; check `lane_stale_slow` in `nav.csv` and measure how far past the lane's end the robot rolls.
- **Stop line timing:** the robot reaches a stop line `STOP_DELAY_MS` (1.5 s, `src/navigation/stop_line.py`) after it leaves the view. If it stops short or long of the line, tune that.
- **Turns:** TBD; the crossing goes straight only, whatever the route says (logged `TBD`).
- **Intersection count:** a missed or false stop line shifts the route; check `step` in `nav.csv`.
- **Stop sign and traffic light:** their gates are uncalibrated, so both can be missed or falsely seen.
- **Gains:** Ignacio's bench values. His normalized-offset path assumes 30 cm per unit of `lane_offset`, which isn't measured; pass `--cm-per-px` once the ground scale is known and the navigator steers by cm.

| Symptom | Likely cause |
| --- | --- |
| Robot steers the wrong way | Check the lane offset sign in `p3.csv` against the video before touching the navigator |
| `braked ... contract N` | A navigator bug; `nav.csv`'s `event` column says which rule |
| FPS well under 20 | Check `phase2_ms` in `nav.csv`; `--no-display` so the render doesn't run afterwards |
| Robot never moves | `motors OFF` in the summary (replay or `--no-motors`), or it braked from the first frame: check `reason` |
