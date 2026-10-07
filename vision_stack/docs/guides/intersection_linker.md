# Intersection Linker

> One intersection per run, straight, left or right, with the whole chain driving. Each run is recorded and judged PASS or CHECK.

`intersection_linker` proves the intersection sequences on the mat before a full course. Each sequence is a `navigation_linker` run with a one-step route. The robot drives exactly as it would in a course run: camera → Phase 2 → Phase 3 → `Navigation` → motors. It approaches a real stop line, crosses or turns, and the run ends once lane keeping has held the lane for 1 s afterwards. Everything is recorded, and the video is rendered after each run.

**Code:** `src/intersection_linker.py` · **Rig:** `src/linker_io.py` · **The sequences:** `src/navigation/intersection.py` · **Tests:** `src/tests/test_intersection_linker.py`, `test_intersection.py`

---

## 1. What a sequence is

The intersection rule drives three stages from the moment the stop line leaves the camera's view:

| Stage | What the robot does | Ends |
| --- | --- | --- |
| `to_line` | Straight to the line, holding its heading on the gyro | At the line, `STOP_DELAY_MS` (0.5 s) after the line left the view |
| `turn` (left/right only) | Left: `(0.36, 0.63)`, a wide arc into the far lane. Right: `(0.45, 0.0)`, a pivot on the right wheel | The gyro reads 85° turned that way (`TURN_TARGET_DEG`), or the time limit (left 4.1 s, right 2.4 s) |
| `exit` | Straight on the new heading | Both lane lines for 3 frames, one for 6, or 3 s; then lane keeping |

A stop sign or red light at the line stops the robot first; the turn starts after.

---

## 2. Running it

`sudo pigpiod` once per boot. Place the robot in its lane, pointing along it, with the stop line ahead and in view.

```
python3 -m src.intersection_linker left --camera              # one left turn; start button
python3 -m src.intersection_linker all --camera               # straight, left, right: reposition between them
python3 -m src.intersection_linker right --camera --no-button # Enter starts it instead
python3 -m src.intersection_linker straight --camera --no-motors   # bench: nothing moves
python3 -m src.intersection_linker left --video runs/<run>.avi     # replay one intersection
```

Each sequence stops by itself. The backstop is `--max-run-s` (20 s), and Ctrl-C stops the motors first. `--camera-control KEY=VALUE` sets a camera exposure or white-balance control for the run (repeatable; `phase1_capture.md`).

---

## 3. Reading the results

Everything goes to `runs/intersection_<time>/`. There's one folder per maneuver (a full `navigation_linker` run folder, plus `sequence.json`), and `summary.txt` / `report.json` beside them:

```
[LEFT] PASS   ended by sequence done (lane held after the crossing)
  stages        to the line 1.52 s, turn 1.48 s, exit 0.20 s
  intersections 1
  turn          ended on gyro target, at -85.3 deg
  heading       -91.2 deg turned, expected -90
```

**PASS needs all of these:**

| Check | A CHECK means |
| --- | --- |
| Exactly 1 intersection counted | The line was missed, or something else passed for a stop line |
| A turn ended on the **gyro target** | It ran to its time limit: the IMU isn't reading, or `IMU_YAW_SIGN` is wrong |
| The lane was held after the crossing | The robot ended off its lane: the run hit the cap or the end of course |
| Heading within 20° of −90 / +90 / 0 | The turn is too short or too long: tune below |
| No command braked for the contract | A navigator bug |

**What to tune from the numbers:**
- **Turns too far** (heading past ±90 by a lot): lower `TURN_TARGET_DEG`. The robot coasts after the turn stops.
- **Turns too short:** raise it.
- **The turn starts too early or late at the line:** `STOP_DELAY_MS` in `src/navigation/stop_line.py`. It also moves where the robot stops at signs.
- **The arc is too wide or too tight:** the duties `LEFT_TURN` / `RIGHT_TURN` in `src/navigation/intersection.py`.

To see why a sequence went wrong, use `nav.csv` in that maneuver's folder. In the turn's rows, `rule` is `intersection` and `stage` is `turn`, `heading_deg` climbs to the target, and `turn_end` says how it ended. `nav.avi` shows the same, with `[intersection / … / turn]` in the strip.
