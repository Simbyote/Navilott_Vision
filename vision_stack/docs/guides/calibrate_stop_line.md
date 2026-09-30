# Stop-Line Calibration (Tape Marks)

> The stop line's distance in cm from a few strips of tape, no checkerboard.

The robot brakes when the stop line is close enough (`LaneKeepingNavigator`: 3 cm), so the stop line needs a distance in cm. The ground homography (`calibrate_ground.md`) gives one, but it needs a checkerboard found flat and square on the floor, and the far squares are only a few pixels tall, so the board is often not found at all. This calibration skips the board. You lay tape at a few measured distances, the robot's own stop-line detector reports where it sees each one, and a curve through those points turns any stop line into cm.

It writes `calibration/stop_line_table.json`. The pipeline uses it for the stop line's cm (`distance_cm`, `stop_line_distance_cm` in the packet) whenever there's no ground homography. With both, the homography wins.

**Code:** `src/scripts/calibrate_stop_line.py`, `src/perception/stop_line_table.py` · **Tests:** `test_calibrate_stop_line.py`, `test_stop_line_table.py`

---

## How it works

The stop-line detector already reports how many rows above the bottom of the lane ROI the line's near edge is (`distance_px`). On a flat floor seen by a fixed camera, the distance in cm grows with those rows along one curve:

```
cm = A / (B - rows) + C          (B is the horizon, in rows above the ROI bottom)
```

The curve has three numbers, so three marks define it and every extra mark checks the others. The script fits them by least squares and prints each mark's error. The rows come from the same detector the robot drives with, so its quirks are calibrated in.

It gives **distance ahead only**, not left/right. That's all the stop line needs.

---

## Requirements

- The lens calibration done first (`calibrate_camera.md`), as for everything that measures on the undistorted frame.
- The camera mount final. The table is tied to the lens calibration, `undistort_alpha` and the 480×270 output, like the homography. The pipeline refuses it with a warning if any of them changes.
- Tape: the real stop-line tape if you have it, otherwise any strip about as wide, long enough to cross the lane.
- A tape measure.

---

## 1. Pick your reference point

Every distance is measured from **one point on the robot**; the default is its **front edge**. The navigator's 3 cm then means "brake when the line is 3 cm in front of the robot". The point is recorded in the file (`--reference "front bumper"` to name another).

---

## 2. Run it

From `vision_stack/`, with the robot on the mat in its driving pose:

```
python3 -m src.scripts.calibrate_stop_line
```

For each mark:

1. Lay the strip **straight across the lane, square to the robot**, not touching a lane mark (a strip touching a lane mark can merge with it and not be seen as a stop line).
2. Measure from your reference point to the strip's **near edge**.
3. Type that distance in cm and press Enter. The script looks at 15 frames (`--frames N`) and prints what it saw:
   ```
   6 cm -> 48.5 rows (seen in 15/15 frames)
   ```
4. Move the strip to the next distance.

**Good distances:** 4–6 marks, from just past where the camera's view begins out to the farthest you care about. For example 3, 6, 10, 15, 20 cm.
- **Nearer marks matter most,** because that's where the robot brakes.
- **Beyond the farthest mark** the curve extrapolates, and the script says so.

**At the prompt:**
- **Enter** on an empty line fits once you have 3 or more marks.
- **`u`** undoes the last mark.
- **A mark that wasn't seen isn't recorded,** and the script says why: the line wasn't found in enough frames, or it's under the bottom of the view.

---

## 3. Check the result

It prints each mark, the curve's value there and the error:

```
   mark cm    rows  fit cm  error
      3.00     6.2    3.04   0.04
      6.00    24.9    5.93   0.07
     ...
  worst error 0.09 cm, mean 0.05 cm -> GOOD
  view bottom reads 2.4 cm; the farthest mark (66 rows) 20.1 cm
```

| Output | Pass |
| --- | --- |
| verdict | GOOD (worst < 0.5 cm) or OK (< 1.0 cm) |
| one mark far off the others | That mark was mismeasured or the strip wasn't square: redo it (`u`, then re-enter) |
| `view bottom reads` | About the distance from your reference to where the camera's view of the floor begins |
| a warning with only 3 marks | Add a fourth: three marks always fit exactly, so nothing checks them |

**Then check one new distance.** Lay the strip somewhere you didn't calibrate and run `python3 -m src.phase3_linker --camera`. The status line's `line=<px>/<cm>cm` should read about what the tape measure says.

---

## Refit without the camera

The marks are saved in the file. To refit from them, or from `line=<px>` values read off `phase3_linker`, pass them as `cm:rows`:

```
python3 -m src.scripts.calibrate_stop_line --marks 3:6.2,6:24.9,10:41.5,15:55.0,20:66.3
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `seen in only N of 15 frames` | The whole strip must cross the lane in view, straight, lit, and not touching a lane mark. Try the real stop-line tape |
| `off the bottom of the view` | The strip is under the camera's view; move it further out |
| `a mark higher in the image must be farther away` | Two distances were swapped or mistyped; `u` and re-enter |
| `don't fit a flat floor's curve` | A mismeasured mark, or the mat isn't flat where the strip lies |
| POOR | Re-measure the worst mark; check the strip lies flat and square |
| `stop_line_table.json not used (...)` at startup | The camera setup changed since the fit (the reason is in the warning); redo it |

**Redo it** after changing the camera mount, height or angle, `undistort_alpha`, the output size, or the lens calibration.
