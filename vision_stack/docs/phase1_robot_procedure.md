# Phase 1 on the Robot: Lane Offset and Lane Keeping

> The robot-side steps that close Phase 1 of `resolution_plan.md`. The software (F1 foot point, F2 full-width lane ROI, F3 steering in cm) is on `claude/wonderful-pasteur-htu1fa` since 2026-11-12; nothing here changes code except the values step 1 measures.

**Time:** about 1.5 h with two people. **Needs:** the practice arena or any straight lane ≥ 1.5 m, a ruler, a charged pack (≥ 11.5 V at rest), someone to walk beside the robot in steps 3–4.

**Order matters.** Step 1 sets the values every later step reads. Don't drive (steps 3–4) before step 2 passes: since F3 the robot steers about twice as hard at 1–4 cm off centre as it did on 5 Nov, and step 3 is the check that this is safe.

---

## Step 0: Session

```bash
make session TIME="YYYY-MM-DD HH:MM"
```

```bash
cd ~/Navilott_Vision && git fetch origin claude/wonderful-pasteur-htu1fa && git checkout claude/wonderful-pasteur-htu1fa && git pull
```

```bash
cd vision_stack && make test
```

`make test` must end with no failures on the Pi (its OpenCV 4 passes the calibration tests the review machine could not run). Note the commit (`git log --oneline -1`) on the results sheet below.

---

## Step 1: Calibrate the lane in the new ROI (`make calib-lane`)

**Why:** F2 widened the lane ROI to the frame's whole width and moved `robot_x_px` from 224 to 248 by arithmetic. This measures it. The ground scale (`config.CM_PER_PX`) follows `expected_half_lane_px`, so it is set by the same edit.

1. Park the robot centred in a straight lane, pointing along it, both lines in view. Check centred with the ruler: equal distance from the robot's centreline to each line's centre.
2. Run:

   ```bash
   make calib-lane
   ```

   If you couldn't centre it exactly, measure how far right of centre it sits and pass it:

   ```bash
   make calib-lane ARGS="--offset-cm 0.5"
   ```

3. It prints three lines for `config.py`:

   ```
   for config.py, MEASURED's LaneOffsetConfig:
           robot_x_px = ...,
           expected_half_lane_px = ...,      # now 197.1
           min_lane_width_px = ...,          # now 260
   ```

**Accept when:**

| Reading | Expected | If not |
| --- | --- | --- |
| `robot_x_px` | 240–256 (arithmetic gave 248) | Outside: the mount moved since 5 Nov. Set it anyway; note it |
| lane width | 380–410 px (2 × 197.1 = 394) | Outside: the camera's height or tilt changed. Set it; step 2 will judge it |
| warnings | none about "both lines" or "the lane middle moved" | Re-park and run again |

4. In `src/config.py`, set **`robot_x_px`** and **`expected_half_lane_px`** to the printed values. Leave `min_lane_width_px` at 260 unless the lane width came out under ~290 px (then use the printed value). Don't touch `max_lane_width_px` (450).
5. Run the settings checks, then commit:

   ```bash
   make test ARGS="src/tests/test_settings.py src/tests/test_config.py"
   ```

   `test_config` pins the old numbers and will fail on the edit: update its two pinned values to the ones you set, and its comment's date. `test_settings` must pass unchanged: it checks the lane fits its ROI and its width gates.

   ```bash
   git commit -am "calib-lane on the robot: robot_x_px <value>, expected_half_lane_px <value> (YYYY-MM-DD)"
   ```

---

## Step 2: P3, lane offset against the ruler (`make routine-lane-offset`)

**Why:** P3's verdict, now judged on both its rules: every placement within ±2 cm, **and** the best-fit slope within 0.8–1.2 (the routine prints both since T1). On 3 Nov the slope was 0.63; the software change predicts ~0.96.

Run it on **two different straights** (P3's pass rule asks for two):

```bash
make routine-lane-offset
```

It prompts each placement: 0, +2, −2, +4, −4 cm (+ = robot **right** of the lane's centre). At each, measure with the ruler from the lane's centre (halfway between the two lines' centres) to the robot's centreline, square to the lane, and type what you measured, not the planned value. The 0 cm look comes first and sets the routine's own scale; at the end it prints that scale beside the one the robot runs (`config.CM_PER_PX`). They should agree within ~2% after step 1.

**Pass:** the routine prints `PASS` with both criteria:

```
PASS  worst |reported - measured|: ... cm (needs <= 2 cm)
PASS  slope of reported against measured: ... (needs 0.8..1.2)
```

on both straights.

**If it fails:**

| Symptom | Likely cause | Next |
| --- | --- | --- |
| Slope still ~0.6 | The robot is running an old checkout, or `robot_x_px`/ROI not updated | `git log --oneline -3`; confirm `roi_crop.LANE` is x 0.00–1.00 |
| Slope ≥ 0.8, one placement over 2 cm, all on one side | `robot_x_px` off: the error is a constant | Redo step 1 more carefully centred |
| A placement reads `vision_pct` well under 100 | One line lost at that placement | Replay that look: `make phase2 FRAMES=runs/routine_lane-offset_<time> VIEWS=lanegeo` and see which gate refused the line |
| Slope over 1.2 | Extrapolation overshooting on real tape | Same lanegeo replay; send the folder back for review |

Then generate the figures for the record:

```bash
make routine-report RUN=runs/routine_lane-offset_<time>
```

---

## Step 3: Recovery and holding a straight (`make routine-recover-offset`)

**Why:** the first time the robot drives with the cm gains. They reproduce 5 Nov's steering near centre, but read the true offset, so they steer 1.8–2× harder at 1–4 cm off. This checks that it recovers without swinging past the far line.

**Motors are on. One person walks beside the robot, ready to lift it.**

```bash
make routine-recover-offset
```

It prompts each start: 0 (a 5 s straight run), then ±2, ±3, ±4, ±5 cm. Place the robot at the offset, **pointing along the lane** (not angled back toward the centre), type the measured start, press Enter; it drives until it has held the centre 1 s or 6 s pass. Then measure where it stopped, the same way.

**Pass:** the routine's `PASS`:

```
PASS  starts up to 3 cm not recovered: 0 (needs <= 0)
PASS  straight run: farthest from the center: ... cm (needs <= 1.5 cm)
```

**Also read** the `overshoot_cm` column for every start of ±3 cm and beyond. That number decides whether to re-tune:

| `overshoot_cm` at ±3–4 cm | Meaning | Action |
| --- | --- | --- |
| ≤ 1.0 | Settles without swinging | Keep the gains. Go to step 4 |
| 1.0–2.5 | Swings past centre, comes back | Halve the gain's growth: `GAIN_SCALE = 0.1` in `src/navigation/lane_keeping.py`; rerun step 3 |
| > 2.5, or crosses the far line | Too strong | `GAIN_SCALE = 0.1` **and** `KP_CM = 0.030`; rerun step 3 |

Any gain change: update the pinned values in `src/tests/test_lane_keeping.py` (`test_the_gains_the_robot_drives_with`); the "as the robot did until 2026-11-12" test is the old behaviour's record and will fail by design if `KP_CM` or `GAIN_SCALE` changes: mark it with the date it stopped applying rather than deleting it. Commit with the overshoot numbers in the message.

```bash
make routine-report RUN=runs/routine_recover-offset_<time>
```

---

## Step 4: One lap with the whole chain (`make navigate`)

**Why:** lane keeping inside the full chain, with intersections, against 5 Nov's run.

**Motors on, someone walking beside it.** The route is `route.json`.

```bash
make navigate
```

Then:

```bash
make nav-report
```

```bash
make timeline
```

**Compare with the 5 Nov course run's `nav-report`** (lane-keeping block):

| Line in `nav-report` | 5 Nov | Accept |
| --- | --- | --- |
| lane keeping: `vision` share | (from that run's report) | Not lower |
| steer histogram | (from that run's report) | No pile-up at the ±0.40 clamp |
| findings | (from that run's report) | No new "runs right/left of center" bias finding; no new `lane_stale_slow` stretches |
| summary.txt `[LANE]` / `lane_offset_cm` column | blank on 5 Nov | **Filled** on every vision frame (F3 working) |
| `source` column in `nav.csv` | `offset` | **`offset_cm`** |

If the robot weaves on straights after intersections (the exit stage hands over to lane keeping), that is the stronger gain meeting a large offset; go back to step 3's table one row down.

---

## Optional: Confirm F1's mechanism on 3 Nov's looks

If the 3 Nov `routine_lane-offset` folders are still on the Pi, replay one on the code from before F1 (commit `b30df40`) and on the current one, and compare the lanegeo videos. Commit step 1's edit first, so the switch is clean.

```bash
git switch --detach b30df40 && make phase2 FRAMES=runs/routine_lane-offset_<3 Nov time> VIEWS=lanegeo ARGS="--no-display"
```

```bash
git switch claude/wonderful-pasteur-htu1fa && make phase2 FRAMES=runs/routine_lane-offset_<3 Nov time> VIEWS=lanegeo ARGS="--no-display"
```

Each writes its own `runs/` folder. At the ±2 and ±4 cm looks, the old run shows one anchor's foot at the ROI's side edge; the new one carries it past the edge. Copy both folders off the Pi for the review if anything looks different.

---

## Results sheet

Fill in, commit as `docs/phase1_results_<date>.md` or paste into `requirements.md`'s P3 status.

| | Value |
| --- | --- |
| Date, commit | |
| Pack at rest, start / end | |
| **Step 1** `robot_x_px` / lane width px / `CM_PER_PX` | / / |
| **Step 2** straight A: worst error / slope / PASS-FAIL | / / |
| **Step 2** straight B: worst error / slope / PASS-FAIL | / / |
| **Step 3** recovered up to / straight run max / worst overshoot / PASS-FAIL | / / / |
| **Step 3** gains after (if changed) | `KP_CM` / `GAIN_SCALE` |
| **Step 4** run folder; `vision` share; new findings | |
| Run folders | |

**P3 is Met** when step 2 passes on both straights; update `requirements.md` P3's status with the numbers and the commit, and its "Changes to pass rules" table needs no entry (the rule didn't change).
