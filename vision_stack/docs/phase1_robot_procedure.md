# Phases 1 and 2 on the Robot: Lane Offset, Lane Keeping, the Frame Budget and the Stop Sign

> The robot-side steps that close Phase 1 of `resolution_plan.md`, then Phase 2's frame budget and a course run that checks F6 (steps 5–6), and the stop-line table's re-fit with D4b and D6 (step 7). The software (F1 foot point, F2 full-width lane ROI, F3 steering in cm) is on `claude/wonderful-pasteur-htu1fa` since 2026-11-12; nothing here changes code except the values step 1 measures.

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

## Step 5: Phase 2, the frame budget (bench, motors off; ~30 min)

**Why:** F4 (the stop sign on the side thread) and F4b (one gradient pass for both splits) should take ~3 and ~4 ms off the main thread. From the 28 Oct and 4 Nov runs: Phase 2 median ~39.5 ms, `total` p95 52.2 ms; the expectation is ~33 and ~45. Nothing drives, so this can run before steps 3–4.

```bash
make routine-frame-budget
```

Read each attempt's `summary.txt` `[TIMING]` block and compare with the 4 Nov run's (below). The geometry parts no longer include `sign`; it shows as `side_sign`.

| Line | 4 Nov (driving) | Expect |
| --- | --- | --- |
| `geometry` median | 26.4 ms | ~19–20 |
| `geometry.edges` | 5.8 | ~5 (one split, not two) |
| `geometry.lanes` | 14.0 | ~11–12 (no split of its own) |
| `side_sign` | (inside geometry, 2.9) | ~3, on the side thread |
| `branch_wait` | 0.04 | still near 0: the side thread (~16 ms) shorter than the main path |
| `TOTAL (median)` | 53.1 | ~46 |

**Pass (the bench diagnostic, not P1/P2):** frame-budget's own `PASS`. If `geometry` hasn't dropped by at least 4 ms, send the `summary.txt` files back: the shares on the Pi differ from the review machine's.

---

## Step 6: One course run with F6, then the replay (motors on; 10 min + 10 min of analysis)

**Why:** F6 (a stop sign counts after 6 votes in one sighting) removed every ghost sign stop and kept every real one when replayed over 28 Oct and 4 Nov; a real run confirms it, and gives P1/P2 their driving numbers. **Place every stop sign before the lap starts:** on 4 Nov the lap-1 "miss" was a sign set down as the robot arrived.

```bash
make routine-figure-eight
```

Then, with the run's folder:

```bash
make nav-replay RUN=runs/routine_figure-eight_<time>/attempt_01
```

```bash
make approach-log RUN=runs/routine_figure-eight_<time> ARGS="--lap-len 14"
```

| Check | Where | Pass |
| --- | --- | --- |
| Every sign in place stopped for | `approach-log`: positions with a sign, `stopped_for` = stop sign every lap | All (D2) |
| False sign stops | `approach-log`: `stopped_for` = stop sign at a position without one | ≤ 1 per 10 min (D2); expected 0 |
| Real signs' votes | `approach-log`'s `sign_frames` at sign positions | All ≥ 6 (F6's threshold); report the lowest |
| The replay agrees with the robot | `nav-replay`: `0 removed, 0 added` | Today's code drove it, so it should; anything else is a replay limit to report |
| P1, P2 | the attempt's `summary.txt` | ≥ 20.0 FPS effective, ≤ 5% over 50 ms; `total` p95 ≤ 50 ms |
| D1 (agreed 2026-11-12) | the routine's own criterion, "lane on vision outside crossings (D1)" | ≥ 90% |

Keep the run folder **with its frames** for the review: `nav-replay` and `approach-log` need only `nav.csv` and `p3.csv`, but a ghost or a miss needs its frames.

---

## Step 7: Re-fit the stop-line table, then D4b and D6 (bench, ~1 h)

**Why:** the stop line read 1.7 cm long on 2 Nov (D4b), and the team chose to re-fit the table rather than shift it (decision 8). The obstacle's cm come from **the same curve** (there is no ground homography), and D6 passed with the old one, so D6 is measured again after.

Keep the old table for comparison:

```bash
cp calibration/stop_line_table.json calibration/stop_line_table_20261028.json
```

1. **Fit.** Lay a strip of the course's stop-line tape across a straight lane, square to the robot, and measure from the **robot's front edge** to the tape's near edge (the reference D4b and D6 use). Run:

   ```bash
   make calib-stopline
   ```

   At each prompt, type the distance you measured. Use 5–6 marks from where the line first shows at the view's bottom out to ~20 cm, for example 9, 11, 13, 15, 17, 20. It prints each mark's error and a verdict: **GOOD** (worst under 0.5 cm) or **OK** (under 1.0) is acceptable; **POOR** means remeasure a mark.

2. **D4b, the stop line's distance.** New placements, not the marks just fitted:

   ```bash
   make routine-detect-range ARGS="--set target=line --set gaps=8,10,12,14,16 --set angles=-10,0,10"
   ```

   ```bash
   make line-replay RUN=runs/routine_detect-range_<time>
   ```

   Name the new run's folder: `RUN=runs` would pool it with the 2 Nov runs, measured on the old table.

   **Pass:** every look within **±2 cm** of the tape with a frame-to-frame SD under 0.5 cm (`line-replay` judges D4b's revised rule and counts the original beside it). Its findings print the table's `C` against the median error: after a good re-fit that shift should be under 0.5 cm.

3. **D6, the obstacle's distance with the new curve.** For the yellow cube and the solder box (the classes D6 judges):

   ```bash
   make routine-obstacle-check ARGS="--set object=cube --set gaps=15,20,25 --set tolerance_cm=2.5"
   ```

   ```bash
   make routine-obstacle-check ARGS="--set object=box --set gaps=15,20,25 --set tolerance_cm=2.5"
   ```

   **Pass (D6):** at 15, 20 and 25 cm, each object **seen in ≥ 90% of frames** (the `seen` column: the routine's own floor is 50%, so read it) and the reported distance within **±2.5 cm**; clear-lane trials with no false frames.

If D6 now fails where it passed, the re-fit moved the far rows: compare `stop_line_table.json`'s marks with the old file's, and send both back.

Commit the new table with the fit's verdict in the message.

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
| **Step 5** geometry / side_sign / TOTAL median | / / |
| **Step 6** signs stopped for / false sign stops / lowest real votes | / / |
| **Step 6** FPS / over budget / total p95 | / / |
| **Step 7** fit verdict / D4b looks within ±2 cm / median error / D6 cube, box | / / / , |
| Run folders | |

**P3 is Met** when step 2 passes on both straights; update `requirements.md` P3's status with the numbers and the commit, and its "Changes to pass rules" table needs no entry (the rule didn't change).
