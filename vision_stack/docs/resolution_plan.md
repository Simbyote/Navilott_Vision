# Resolution Plan: from the 2–5 Nov results to a system that meets its requirements

> A review of the software architecture, what each requirement's check actually found, the proposed fixes, and the order to integrate them. Draft for team review, 2026-11-12. Nothing here is applied yet.

Sources: `architecture.md`, `requirements.md`, `course.md`, the four `vision_pipeline/` docs, and the code under `src/` at commit 8691962. Numbers quoted are from the docs' status columns (bench runs 2–5 Nov, course run 2026-11-05) unless marked **new**, which means they were derived in this review from the code and the recorded calibration values.

---

## 1. Architecture review

### What holds up

| Property | Where it shows | Why it matters here |
| --- | --- | --- |
| Seven layers, imports point down only | `architecture.md` "Software layers"; `test_navigation_contract` loads the stages with pigpio blocked | Phases 2–4 run on a laptop and in replays, so every fix below can be checked on recorded frames before the robot moves |
| One data type per hand-off | `FrameData`, `SensorSample`, `Phase2Output`, `EstimationPacket`, `Command` | A fix in one stage can't silently change another's input shape; `Phase2Output` validates stamps at construction |
| Memory only in Phase 3 | `estimation.py`: filter, votes, holds, odometry | Phase 2 is testable one frame at a time; the parked routines and `make eval` score it directly |
| Production / debug twins held equal by tests | `estimate_*` beside `compute_*`, `Pipeline` beside the linkers, `test_production_parity`, `test_pipeline` | The robot never pays for overlays, and a tuning change is seen in the linker's video with the same numbers |
| Tuning in one place, pinned by tests | `config.py` `MEASURED` / `MEASURED_ESTIMATION`; `test_settings`, `test_config` | A calibration is an edit plus a test update, and the test says what the robot drives with |
| Safety independent of Python | Watchdog copy inside pigpiod; `main.run()` halts in `finally`; wheel-stall guard; battery CRITICAL latch | The failure modes that damage the robot are covered before any of the tuning work |
| One entry point with a self-test | `Makefile` (`make help`), `test_makefile` | New criteria and tools slot in without a second way to run things |
| Every requirement has a routine or analysis that judges it | `routines/`, `analysis/`, the IADT table | The verification campaign in §6 needs no new measurement method, only runs |

### Weak points that the failing requirements trace back to

1. **The robot steers on a scale the docs say it doesn't use (new).** `MEASURED_ESTIMATION` leaves `cm_per_px` unset (`config.py`; `course.md` "Ground scale: Not set"), so on every run `lane_offset_cm` is `None` and `LaneKeepingNavigator` takes the normalized path: `offset × KP_NORM × (1 + GAIN_SCALE × |offset| × NORM_TO_CM)` with `NORM_TO_CM = 30`, a value the code itself marks unmeasured. One unit of `lane_offset` is half the lane ROI, 216 px, which at the measured 394 px / 14 cm lane is **7.7 cm, not 30**. The gain growth term is therefore ~4× what was intended, and P3 measures a quantity (`lane_offset_cm`) the robot never computes while driving. `navigation_contract.md` describes lane keeping as steering "by `lane_offset_cm`". This is a config-vs-reality gap, not a bug in either module.

2. **ROI geometry isn't derived from the course facts (new).** The lane ROI spans x 5–95% of the frame, 432 px. The lane at the ROI bottom measures 394 px, centred at `robot_x_px` 224: it sits at 27–421 px, leaving **27 px (0.96 cm) on the left and 11 px (0.39 cm) on the right**. `course.md`'s "Values that come from these measurements" table has no entry for this margin. Consequences in §2 (P3).

3. **Foot-point measurement assumes the line reaches the ROI bottom.** `contour_foot_x` takes the mean x of the contour's lowest 6 rows. A line that leaves the ROI through its side has its lowest rows at that side, so its foot reads as the ROI edge. Combined with item 2, every placement beyond ~0.4–1 cm off centre clamps one foot.

4. **Phase 2's critical path carries work that doesn't belong on it.** `detect_geometry` runs edges → lanes → stop line → **sign** on the main thread; the sign detector reads only the colour sign ROI and never touches the Canny map, yet the parallel split (`perception/parallel.py`) was drawn by branch (geometry vs colour/obstacle), not by dependency. Geometry is 26.3 ms of the 40.1 ms Phase 2 median; the side thread finishes in ~11 ms and the main thread waits on nothing.

5. **One persistence rule for four detectors.** `Phase3Config.vote_window = 3` (2 of 3) is shared by the light, sign, stop line and obstacle. A real sign is in view for 25–60 cm of approach at ~8 cm/s, i.e. tens of frames; the three false signs on 5 Nov each came from 1–3 frames beside a red lamp. The sign's persistence requirement should be set from how the sign is actually seen, not from the light's.

6. **Navigation constants live at module top, not in config.** `REACH_CM`, `SIGN_MEMORY_MS`, `TILT_TO_HEADING`, the turn duties, etc. are module constants in `src/navigation/`. Tests pin them, which works, but a navigation re-tune is a code edit rather than a `MEASURED` edit like Phase 2's. Not a cause of any failure; noted because Phase 1 below changes lane-keeping gains and should put them where tuning lives.

7. **Twin logic is duplicated by hand.** `_usable` / `_is_usable`, `_single_sided` / `_project_single`, `compute_*` / `estimate_*`. Parity tests catch divergence, but every gate change is two edits. The fixes below touch `lane_offset.py`; both twins must change together.

8. **Docs state a status the tables contradict (new).** `requirements.md` "Status summary" lists P1, P2, P3, D2, D6 under *Verified* and R5 as "software + hardware", while their rows say *Not met* (P1, P2, P3, D2, D4b, D6 for the blue can) and R5's hardware test is "written, not yet recorded". `README.md` "Current status" says every requirement is verified except D1. *Verified* in this document means "the check has run", which is true, but the summary reads as "met". It must say which.

9. **P1's pass rule can't be met as written (new).** The camera is set to 20 FPS and the appsink drops a frame whenever processing overruns, so effective FPS = 20 × (1 − dropped share). The same rule allows ≤ 5% of frames over 50 ms, which alone allows ~19 FPS. "≥ 20.0 effective" is attainable only with ~0 over-budget frames. The two clauses need reconciling (see Decisions, §5).

---

## 2. Requirement status, with the cause as far as it is known

| ID | Verdict (5 Nov rule) | Gap | Cause | Confidence |
| --- | --- | --- | --- | --- |
| P1 | Not met | 19.2 FPS vs ≥ 20.0; over-budget share 2.8% (≤ 5% met) | Processing, not capture (`capture-anatomy`: 49.996 ms period, 0 lost). Phase 2 median 40.1 ms; the p95 tail crosses 50 ms and drops the next frame | High |
| P2 | Not met | p95 52.2 ms vs 50; median 43.0 | Same as P1: geometry 26.3 ms + preprocess 12.0 ms on the main thread | High |
| P3 | Not met | slope 0.63 vs 0.8–1.2; 19/32 within ±2 cm | **New:** one lane line's foot clamps at the ROI side beyond ~0.4–1 cm off centre (§1 items 2–3). The model predicts a two-boundary width of 292–308 px and a slope of 0.55–0.74 at ±2–4 cm; the field read 270–300 px and 0.63. The single-line runs that read "up to 4 cm wrong" are the far-side cases where the clamped line still counted | High for the mechanism; verify on the saved looks (§4, F1) |
| P4 | Met by design | | | |
| D1 | Not verified | Target not agreed; 10-min run: `vision` 75% whole-run incl. intersections | Open decision #1 | n/a |
| D2 | Not met | 4/5 signs confirmed; 3 false per 9.3 min vs ≤ 1 per 10 | False: a red traffic lamp in the sign ROI passes the shape gates for 1–3 frames, and 2 of 3 confirms it. Missed (lap 1, int. 6): unknown; the approach is in the run's frames | High (false), Low (missed) |
| D3 | Met, one lighting setup | Second setup owed; driving: 6/18 unlit lights decided red/yellow beside a lit one | The traffic ROI admits a neighbouring intersection's lamp; nothing ties a lamp to the line being approached | High |
| D4a | Met (revised) | Plain-lane 300-frame check owed | | |
| D4b | Not met | +1.7 cm median; 34/57 within ±2 | A steady offset after the 28 Oct table; the table's own fit error was 0.2 cm, so the mount or reference moved after it was fit. The curve's `C` term is additive, so the correction is one number | High |
| D5 | One run of ≥ 2 | 44/44 crossings (28 Oct) | | |
| D6 | Met (cube, box); not met (blue can 0/9) | Can seen only at 10 cm | Dark and narrow: which gate refuses it isn't yet named (`make obstacle-map` does that) | Medium |
| R1–R3 | Verified in software | | | |
| R4 | Verified while driving (2.2 min; memory growth needs 5) | | | |
| R5 | Verified in software; hardware test written, not recorded | | | |

---

## 3. Proposed resolutions

Each fix names the requirement it serves, what changes, where, how big, how it is verified, and what it could break. Sizes: **S** under ~30 lines in one or two files; **M** under ~150 lines across a few files plus tests; **L** more, or a behaviour change needing mat time.

### 3.1 Code changes

| # | For | Change | Files | Size | Verified by | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| **F1** ✅ software | P3 | **Foot point by extrapolation.** Fit the contour's axis (`cv2.fitLine`, or the `minAreaRect` axis the gates already compute) and evaluate x at the ROI bottom row, instead of the lowest band's mean x. A line that leaves through the side then reports where it *would* meet the bottom, inside or outside the ROI. Keep `proximity` as it is. Apply in both twins (`geometry_lanes` stores `foot_x`; `lane_offset.foot_x` is the fallback). *Implemented 2026-11-12; see Phase 1, step 1* | `perception/geometry_shared.py` (`contour_foot_x`), `geometry_lanes.py`, `lane_offset.py` | M | Replay the lane-offset routine's saved looks (`attempt_NN_±Ncm.jpg`) with `make phase2 FRAMES=<dir> VIEWS=lanegeo`: the two-boundary width should read ~394 px at every placement. Then re-run `make routine-lane-offset`. `test_lane_offset`, `test_geometry`, `test_production_parity` | Two-boundary widths off centre will rise from ~290 to ~394 px, close to `max_lane_width_px` 400: raise the cap to ~450 (a full street is ~590 px at the bottom row) in the same change. The `min_lane_width_px` floor of 260 was set to admit the clamped widths; it can stay |
| **F2** ✅ software | P3, lane keeping | **Widen the lane ROI to x 0–100%.** Undistortion with `alpha = 0` already crops the black border, so the extra 48 px are real floor. Gives 51 / 35 px of margin instead of 27 / 11 | `perception/roi_crop.py` `LANE`, `robot_x_px` + 24 (`config.py`), tests pinning the rect | S | `test_roi_crop`, `test_config`; `make calib-lane` once on the robot to re-measure `robot_x_px` in the new ROI | Changes `lane_roi_width_px` (432 → 480) and so the normalized offset's scale; do together with F3 so the gain is re-derived once |
| **F3** ✅ software | P3, lane keeping | **Set the ground scale and port the gain.** `cm_per_px = 14 / lane_width_px` from `make calib-lane` (0.0355 at 394 px) in `MEASURED_ESTIMATION`, so `lane_offset_cm` exists on every run and lane keeping takes the cm path. Port today's effective gain so behaviour doesn't jump: with the normalized path, 1 unit = 7.67 cm, so `KP_CM = 0.30 / 7.67 ≈ 0.039` and `GAIN_SCALE = 0.05 × 30 / 7.67 ≈ 0.196` reproduce the 5 Nov steering exactly. Delete or correct `NORM_TO_CM` (7.7) so the fallback path can't diverge again | `config.py`, `navigation/lane_keeping.py`, `test_settings` (a new invariant: `cm_per_px` set, and the two paths agree at 2 cm) | S | `make routine-recover-offset` (exists) on the mat, then `make nav-dry`/`navigate` on a straight; `nav-report` lane-keeping block | After F1 the offset itself reads ~1.6× larger off centre, so the robot will correct harder than on 5 Nov. Expect one re-tune of `KP_CM` on the mat; that is the point of doing F1–F3 together and verifying with recover-offset before any course run |
| **F4** ✅ software | P1, P2 | **Move the stop-sign detector to the side thread.** Split `detect_geometry` into edges + lanes + stop line (main) and `find_sign_candidates` (side, beside colour and obstacle); assemble `GeometryBranchResult` after `branch_wait`. The result is identical; only who computes it changes | `pipeline.py` (`_perceive_parallel`), `perception/geometry.py` (a `sign=False` path or a separate function), `test_parallel` | M | `make compare-parallel ARGS="--frames runs/nav_<5 Nov>/frames"` before/after, then `make ab-parallel RUNS=4`; `make timing` on a new run. `test_parallel` holds results equal frame by frame | Gain depends on the sign's share of geometry's 26.3 ms, which `summary.txt`'s `[TIMING]` parts already print: read them first (step 0 in §6). If the sign is ≥ 8 ms, the main path drops under ~18 ms and both P1 and P2 should clear |
| **F5** | P1, P2 (if F4 isn't enough) | **Blur only what is read.** The gray blur feeds the lane and obstacle ROIs (rows ≥ 45%), the colour blur the traffic, sign and obstacle ROIs; both run on the full frame today. Blur the rows actually consumed, or blur per ROI inside `crop_rois`. Undistortion stays whole-frame (the sign ROI starts at row 0) | `perception/preprocess.py`, `roi_crop.py`, the debug twins that draw on `color` | M | `make timing`; `test_preprocess`, `test_roi_crop`, `test_production_parity` | Touches the frames every debug view draws on; do only if F4 leaves P2 over 50 ms |
| **F6** | D2 (false) | **Sign persistence in Navigation.** `StopSignRule` remembers a sign only after it has been voted present for ≥ `SIGN_MIN_FRAMES` (first guess 8, ~0.4 s) within `SIGN_MEMORY_MS`; a 1–3 frame ghost never arms the stop. Alternative in Phase 3 (a longer sign-only vote window) is weaker: a 3-frame ghost still passes 3 of 5 | `navigation/stop_sign.py`, `test_stop_sign`, `navigation_checks` (the sign check feeds enough frames) | S | `make approach-log` on the 5 Nov run already gives `sign_frames` per approach: pick the threshold between the real signs' counts and the ghosts' 1–3 before coding, then replay with `make eval ARGS=--from-log`. Then `make routine-stop-sign` | A sign seen late (first at 25 cm, ~3 s before the line) still gives ~60 frames; a threshold of 8 costs nothing there. The risk is the missed sign at lap-1 int. 6, which this doesn't address (see P2 in §3.3) |
| **F7** | D4b | **Stop-line table re-fit, or its `C` shifted.** Re-run `make calib-stopline` on the current mount (the principled fix, ~15 min). Fallback with no camera time: `C −= median error` from `make line-replay` (1.7 cm), since `cm = A/(B − rows) + C`. Add two lines to `line_replay` that print the corrected `C` next to the median error, so the number isn't computed by hand | `calibration/stop_line_table.json`; `analysis/line_replay.py` (print only) | S | `make routine-detect-range ARGS="--set target=line ..."` then `make line-replay` on the new runs: median error within ±0.5 cm, 57/57 within ±2 | None for the print. A shifted `C` is valid only until the mount moves again; the json's `created` date should be updated and the shift noted in its `reference` field |
| **F8** | D6 | **Name the blue can's gate, then decide.** `make obstacle-map RUN=<folder of the 3 Nov runs>` replays each missed look and names the refusing gate. If `narrow`: `min_width_px` can drop from 60 toward 50 (the 47 px opening already removes any tape). If `small`/`short` or the can isn't in the not-floor mask at all, it is a dark object under the agreed limitation and no code changes | `perception/obstacle.py` `ObstacleConfig` (maybe) | S | `make routine-obstacle-check ARGS="--set object=can"`; `test_obstacle` | A looser width gate raises the false-alarm risk; re-run the clear-lane trials (27/27 today) |
| **F9** | D3 (driving) | **Tie the light to its intersection: measure first.** The 5 Nov frames at the six false-wait approaches exist. Label them (`make label`, condition tag `neighbour_lamp`) and run `make eval`; from the colour trace, compare the white-spot size and ROI position of the own lamp at ≤ 20 cm against a neighbour's. Candidate gates, in order of least change: a `min_white_px` for the lamp that counts (a far LED clips fewer pixels), or the traffic ROI narrowed again, or accept a lamp only while the tracker is in APPROACH with the line nearer than N cm. Decide from the data, not before | `perception/color_branch.py` `GlowFilter` or `roi_crop.TRAFFIC`, or `navigation/traffic_light.py` | S–M | `make light-report` on a new course run; the false waits (safe, but each costs ~5 s) should go to 0 of 18 | A gate that is too tight turns a false wait into a run red, which D3 forbids at any distance. Any change here is verified parked at 0–45 cm again before it drives |

### 3.2 Additions to the test and tool set (kept to a minimum)

No new make targets are needed. Three existing commands gain a criterion or a line, and one test file gains invariants:

| # | Where | Addition | Why |
| --- | --- | --- | --- |
| **T1** ✅ | `routines/lane_offset.py` `judge()` | A second criterion: best-fit slope of `reported_cm` against `measured_cm` within 0.8–1.2 (the 5 Nov P3 rule). `analysis/offset_accuracy.py` already computes the fit; call it or copy the two-line least squares | The routine currently judges only the worst error, so it would print PASS on a run the rule fails |
| **T2** | `routines/course_run.py` `judge()` (and `figure_eight.py`) | D1's criteria on lane-keeping frames only: `vision` share ≥ 90% while `rule == lane_keeping`, longest `stale` run ≤ 1.0 s. `analysis/nav_run.lane_keeping()` already computes the shares by rule; the watch can count the longest stale stretch | Once decision #1 is taken, D1 is judged by the same run that judges D2 and D5, with no new tool |
| **T3** | `analysis/line_replay.py` | Print the corrected `C` beside the median error (see F7) | Turns a measurement into the one number to set |
| **T4** ✅ | `tests/test_settings.py` | Invariants between the course facts and the tuning: (a) `MEASURED_ESTIMATION.cm_per_px` is set; (b) at the measured lane width, a ±4 cm placement keeps both feet reconstructable (F1) and the two-boundary width under `max_lane_width_px`; (c) the normalized and cm steering paths agree at 2 cm, or the normalized path is gone | Makes §1 items 1–2 impossible to reintroduce silently |

Everything else in the plan runs on commands that already exist: `calib-lane`, `calib-stopline`, `routine-lane-offset`, `routine-recover-offset`, `routine-detect-range`, `routine-obstacle-check`, `routine-stop-sign`, `routine-figure-eight`, `routine-course-run`, `compare-parallel`, `ab-parallel`, `timing`, `loop-budget`, `approach-log`, `line-replay`, `obstacle-map`, `eval`, `label`, `light-report`, `nav-report`, `diag-run` + `pi-load`, `test-hw -k watchdog`.

### 3.3 Procedures (hardware time, no code)

| # | For | Procedure | Pass when |
| --- | --- | --- | --- |
| **P1** | D5 | A second `make routine-figure-eight` (or `routine-course-run`) run on a different day | Every crossing PASS in both runs |
| **P2** | D2 (missed) | `make approach-log` on the 5 Nov run for lap-1 intersection 6; `make sign-check ARGS="--frames <that approach's frames>"` names the gate that refused the sign, or shows it never entered the red mask (exposure) | Cause named; if it is a gate, a one-value change with `make eval` before/after; if exposure, a `CAM=` setting from `make sweep-camera` |
| **P3** | D3 | The second lighting setup: `make routine-detect-range ARGS="--set target=light"` 0–45 cm | Within 20 cm 100% stop/go right; no go for a lit red or yellow out to 45 cm |
| **P4** | D4a | 300 frames of plain lane: `make phase3 ARGS="--limit 300"` on a straight, then `make stability` | 0 stop-line frames |
| **P5** | R4 | A ≥ 5-minute `make diag-run`, then `make pi-load NAV=<run>` | Peak RSS < 400 MB with no upward trend |
| **P6** | R5 | `make test-hw ARGS="-k watchdog"` with the wheels up | Brake within 0.5 s of the last command, in the artifact's CSV |
| **P7** | P3 | After F1–F3: `make calib-lane`, then `make routine-lane-offset` on two straights | Every placement within ±2 cm and slope 0.8–1.2 (T1 prints both) |
| **P8** | P1, P2 | After F4: a ≥ 5-minute course run; read `summary.txt` | ≥ 20.0 FPS effective (or the reconciled P1 rule, decision #8) and p95 ≤ 50 ms |

---

## 4. Verification of this review's own claims

Three findings above are derived, not measured. Each has a cheap check with existing tools before any code is written:

| Finding | Check | Expected if the finding holds |
| --- | --- | --- |
| **F1's mechanism** (clamped feet) | `make phase2 FRAMES=<routine_lane-offset folder> VIEWS=lanegeo` on the ±2 and ±4 cm looks | The lanegeo view shows one anchor's foot drawn at the ROI's side edge, and `lane_width_px` ≈ 290–310 at ±4 cm |
| **Lane keeping runs the normalized path** | `nav.csv` of any run: the `lane_offset_cm` column is blank and `source` reads `offset`, not `offset_cm` | Confirms §1 item 1 |
| **The sign's share of geometry** | The 5 Nov run's `summary.txt` `[TIMING]` block: the four parts indented under `geometry` | If `sign` ≥ 8 ms, F4 alone should clear P2; if ≤ 3 ms, go straight to F5 |

### Results, 2026-11-12 (run off the robot; the recorded runs live only on the Pi)

The saved looks, `nav.csv` and `summary.txt` were not reachable from the review machine, so each check was run the nearest way the code allows. What still has to be read on the Pi is marked.

| Finding | What was run | Result | Still to confirm on the Pi |
| --- | --- | --- | --- |
| **F1's mechanism** | A lane drawn at the measured geometry (394 px at the ROI bottom, centred on `robot_x_px` 224, converging toward the table's horizon, 10 px tape) through `preprocess_frame` → `crop_rois` → `detect_geometry` → `estimate_lane_offset` under `MEASURED`'s lane config, shifted −4 to +4 cm in 1 cm steps | **Reproduced.** At 0 cm the width reads 399.5 px. Off centre one foot clamps at the ROI side (left foot 5–8 px for +1 to +4 cm; right foot 423–425 px for −1 to −4 cm), the width falls to 273–356 px, and the best fit is **reported = 0.60 × true + 0.14 cm** (field: 0.63 × true − 1.15). With each candidate's axis extrapolated to the bottom row instead (`cv2.fitLine`), the width holds at 414–416 px at every placement and the fit is **1.03 × true − 0.00 cm**. A 20 px tape traced as two edge contours and moved every foot ~22 px inward (the documented wide-tape limit); the slopes were the same (0.57–0.68 against 1.06–1.08) | `make phase2 FRAMES=<routine_lane-offset folder> VIEWS=lanegeo` on the ±2 and ±4 cm looks: one anchor's foot at the ROI's side edge |
| **Lane keeping runs the normalized path** | `Pipeline()` as `src.main` builds it; `LaneKeepingNavigator.update()` on a vision packet with `lane_offset_cm = None` | **Confirmed.** `MEASURED_ESTIMATION.cm_per_px` is `None`, the pipeline's Phase 3 carries none, and the record reads `source: offset`. One normalized unit is 7.68 cm (the code says 30). At 2 cm the normalized path steers 0.109 against the cm path's 0.033: **3.3×** | `nav.csv` of any run: `lane_offset_cm` blank, `source` = `offset` (a formality) |
| **The sign's share of geometry** | `detect_geometry` with `timings` over the 33 synthetic scenes, 30 passes each, on an x86 machine | Shares only (the ms don't transfer): edges 22%, lanes 50%, stop line 6%, **sign 22%**. If the Pi's 26.3 ms splits the same way, F4 moves ~6 ms off the main thread, which takes the Phase 2 median from 40 to ~34 ms and the p95 from 52 under 50. The synthetic sign ROI is nearly empty; the course's has a sign or a lamp in it, so the real share is likely higher | The 5 Nov run's `summary.txt` `[TIMING]` block, the four parts under `geometry`. Decides between F4 alone and F4 + F5 |

**State of the software suite (run for this review, 2026-11-12, off the robot):** 3268 passed, 9 skipped, 31 hardware tests deselected, `make lint` clean. 5 failures and 8 errors, all from the sandbox's OpenCV 5.0: `test_calibration` builds its synthetic boards with `cv2.undistortPointsIter`, removed in OpenCV 5, and one `test_debug_video` check reads exact `putText` colours, whose anti-aliasing changed. The Pi runs the apt OpenCV 4, where both pass. Worth one line in `pyproject.toml` (`opencv-python-headless<5` for laptop installs) so a teammate's fresh venv doesn't read these as regressions.

---

## 5. Decisions the team has to take (before the phase that needs them)

| # | Decision | Needed by | Recommendation |
| --- | --- | --- | --- |
| 1 | D1 target (`vision` ≥ 90% on lane-keeping frames, `stale` ≤ 1 s, flicker ≤ 5% parked) | Phase 4 (T2) | Adopt as proposed; judge on `rule == lane_keeping` frames only |
| 2 | Blue can: dark-object limitation, or a gate change | Phase 3 (F8) | Follow `obstacle-map`'s answer |
| 3 | `SIGN_MIN_FRAMES` value | Phase 3 (F6) | From `approach-log`'s `sign_frames` on the 5 Nov run: halfway between the ghosts' max and the real signs' min, on a log scale |
| 4 | Lane-keeping gain after F1: keep the 5 Nov effective gain (port) or re-tune | Phase 1 (F3) | Port first, re-tune only if recover-offset shows overshoot |
| 5 | Whether the D3 driving case (false wait beside a neighbour lamp) becomes a rule or stays "recorded" | Phase 3 (F9) | Make it a rule only if F9's measurement yields a gate with margin; a false wait is safe |
| 6 | D4b: re-fit the table or shift `C` | Phase 3 (F7) | Re-fit if the robot is available for 15 min; shift otherwise, and re-fit before Design Day |
| 7 | `requirements.md` status summary: rename the columns to *Checked* / *Met* / *Not met* / *Pending* | Phase 0 | Yes; and `README.md`'s "Current status" to match |
| 8 | P1's wording: "≥ 20.0 effective FPS" with a 20 FPS source | Phase 0 | Judge P1 by the over-budget share (≤ 5%, or tighter) and report effective FPS beside it; or set the source to 21 FPS if the sensor mode allows it, which also gives P2's clock 2 ms more. Record in "Changes to pass rules" |

---

## 6. Integration plan, in phases

Each phase ends with a gate: the named check passes on the robot (or on replays where no robot is needed) and the software suite (`make test`, `make lint`) is green. Phases 1 and 2 are independent of each other; 3 depends on 1 only for the course runs that judge it; 4 needs all of them.

### Phase 0: facts straight (no hardware; half a day)

1. `requirements.md`: status summary and README status corrected (decision 7); P1 wording (decision 8) recorded under "Changes to pass rules".
2. The three checks in §4, with their results written into this document.
3. `course.md`: add the lane's lateral margin in the lane ROI to "Values that come from these measurements", with F1/F2 as what depends on it.
4. `navigation_contract.md`: say which offset path lane keeping runs today (the normalized one) until F3 lands.

**Gate:** docs reviewed by one teammate who didn't write them.

### Phase 1: the lane offset is right, and lane keeping steers by it (1 bench session + 1 mat session)

Order matters: measure, then change the scale, then the gain.

1. **F1** foot by extrapolation + `max_lane_width_px` 450. Replay the lane-offset looks (§4) and `make eval` on a labelled nav run: offsets at ±2–4 cm rise, centred ones don't move.

   **Done in software, 2026-11-12.** `contour_foot_x` takes the ROI height; a contour that ends more than half a band (3 rows) above the bottom has its lowest 30 rows' axis (`FOOT_FIT_ROWS`) carried on to the band's row, unless flatter than 20° (`FOOT_MIN_SLOPE_DEG`); contours that reach the bottom keep the band's mean, so centred frames read as before. The measured-geometry test (`test_lane_offset`, the last test) now reads **0.96 × true + 0.02 cm, worst error 0.26 cm**, against 0.60 × true before; the pair's width holds at 374–390 px from −4 to +4 cm. Two things turned up on the way:
   - **The foot's sentinel was −1.0,** and `lane_offset.foot_x` treated every negative stored foot as "not computed" and fell back to the band mean, so the left line's carried-on foot never reached the offset. The sentinel is now `None`; a negative foot is a line past the ROI's left side. The CSV replays and hand-built candidates in the tests use `None`.
   - **A thin synthetic tape sits on two lane gates.** A 6–10 px tape closes into one filled blob of ~1060–1090 px² and elongation ~60–73, against the lane detector's defaults of 1000 px² and 60, so at some placements one line is refused and the one-line projection carries it (within 0.13 cm here). Which placement drops moves with the tape's width and the ROI, so it is not about the ROI border, as first written here. Real tape at 20–40 px traces as two thin edge contours, far under both. Not F1's; noted for the gate re-sweep the docs already owe (`phase2_perception.md`, Stage 4).

   Still owed on the robot: **P7** (`make calib-lane`, then `make routine-lane-offset` on two straights) and the lanegeo replay of the 3 Nov looks. Steps for the robot, with pass rules and what to do on a fail: `phase1_robot_procedure.md`.
2. **F2** lane ROI to full width; `make calib-lane` on the robot for the new `robot_x_px` and widths.

   **Done in software, 2026-11-12.** `roi_crop.LANE` is x 0–100% (480 px); `robot_x_px` 224 → 248, the same frame column, since the ROI's left edge moved 24 px left (`make calib-lane` should read ~248; set what it reads). The lane now has 51 and 35 px to spare, so a line leaves through the side only past ~1.2–1.8 cm off centre and F1's extrapolation runs on fewer frames. Ground homography, stop-line table, stop-line rows and the obstacle ROI are in frame or row coordinates and don't move. Tests: the measured-geometry test reads 0.97 × true + 0.02 cm; the intersection-linker sequence now draws its lane about the ROI's new middle (`scene()`'s default lane, 150–290 px, was drawn about the old one, 216, and read +0.08 off centre), and the short-blob test compares against the clean scene instead of pinning which boundary the blob replaces.

   **One behaviour change rides on it until F3 lands:** the normalized offset is divided by half the ROI's width, now 240 px instead of 216, so the same lateral error reads 10% smaller. Lane keeping's normalized path (the one the robot runs, §4) steers ~10% softer, and Phase 3's jump gate (0.5 normalized) admits 120 px instead of 108. F3 replaces that path with `lane_offset_cm`, which doesn't depend on the ROI's width; **don't drive between F2 and F3** unless the softer steering is wanted for a comparison.
3. **F3** `cm_per_px` in `MEASURED_ESTIMATION`; `KP_CM`/`GAIN_SCALE` ported; `NORM_TO_CM` corrected or removed; **T4** invariants; **T1** slope criterion.

   **Done in software, 2026-11-12.** `config.CM_PER_PX` = 14 cm / (2 × `expected_half_lane_px`) = 0.035515 cm/px, set in `MEASURED_ESTIMATION`, so `make calib-lane` moves the lane width and the scale together; `params.LANE_WIDTH_CM` holds the 14 cm once. `KP_CM` 0.039 and `GAIN_SCALE` 0.195 reproduce the 5 Nov normalized steering to within 1% (a test pins it). The normalized fallback converts with `NORM_TO_CM` 8.52 (was 30) and goes through the same formula, so the two paths can't diverge; `kp_norm` is gone. The three linkers' `--cm-per-px` defaulted to `None`, which would have dropped the scale in every bench run while `make run` used it: they default to `CM_PER_PX` now. **T1:** `routine-lane-offset` judges the slope (0.8–1.2) beside the worst error. **T4:** `test_settings` holds the scale to the lane width, `NORM_TO_CM` to the ROI and scale, and the lane inside its ROI and between the width gates. The pipeline now refuses frames other than 480×270 by default, as it always did with a scale; a linker's `--cm-per-px 0` runs without one, to replay old 480×360 recordings.

   **What the robot will feel:** near centre, the same steering as 5 Nov. At 1, 2, 3 and 4 cm off centre, 1.78, 1.88, 1.96 and 2.02× as much, because the offset now reads the truth instead of ~60% of it. That is decision 4's case: run `make routine-recover-offset` from 4 cm both sides before any course run, and only if it overshoots past the far line, lower `KP_CM` (halving the growth, `GAIN_SCALE` ~0.1, is the gentler first step).
4. `make routine-lane-offset` on two straights (**P7**).
5. `make routine-recover-offset` from 4 cm off centre, both sides: converges without overshoot past the far line. Re-tune `KP_CM` only if it doesn't (decision 4).
6. One `make navigate` lap on the course with `make nav-report`: lane keeping shares and steer histogram no worse than 5 Nov.

**Gate:** P3 PASS with the slope rule; a course lap with no new `lane_stale_slow` stretches.

### Phase 2: the frame fits its budget (1 bench session)

1. Read the geometry parts (§4, third row). 
2. **F4** sign detector on the side thread; `test_parallel` green; `make compare-parallel` on the 5 Nov frames before and after.

   **Done in software, 2026-11-12.** `detect_geometry`, `run_geometry_stage` and `run_geometry_branch` take `sign=False`; in parallel mode `Pipeline` and `run_chain` run `find_sign_candidates` / `extract_sign_candidates` on `p2-side` after colour and obstacle and put its candidates back into geometry's result after `branch_wait`, before fusion. Serial mode is unchanged. Results are identical frame by frame (`test_parallel`, `test_production_parity`); the sign's time shows as `side_sign` (`last_side_ms`, `side_timings_ms`, `parallel_compare`'s side columns) and no longer among geometry's parts. **Expected on the Pi,** from the 5 Nov stage times and the sign's ~22% share of geometry on the synthetic scenes: main path ~20 ms (geometry less the sign, plus lane offset and stop line), side ~17 ms (colour 5.2, obstacle 5.9, sign ~6), so Phase 2's median ~40 → ~34 ms. On the review machine Phase 2 takes under 2 ms on synthetic frames, which is all hand-off overhead and says nothing either way (parallel p50 1.79 → 1.69 ms). The Pi's numbers decide: steps 1 and 3 below.
3. `make ab-parallel RUNS=4 SECONDS=60`, then `make routine-frame-budget` (the bench diagnostic).
4. If Phase 2 median is still over ~35 ms on the bench: **F5** blur only what's read; repeat 3.

**Gate:** bench `frame-budget` PASS on all criteria, and the main thread's `branch_wait` > 0 on most frames (the side thread is now the longer path, or they're balanced).

### Phase 3: detections the course relies on (1 bench session + 1 mat session)

1. **F7** stop-line table (re-fit or `C` shift), **T3**; `make routine-detect-range ARGS="--set target=line ..."` + `make line-replay` (**D4b**). **P4** plain-lane check (**D4a**).
2. **F6** sign persistence, threshold from `approach-log` (decision 3); `make routine-stop-sign` (sign present and absent trials). **P2** names the missed sign's cause and fixes it if it is one value.
3. **F8** blue can: `obstacle-map`, decision 2, `routine-obstacle-check` per object; `make routine-obstacle-stop` (the D6 stop behaviour, not yet tested at all).
4. **F9** measurement on the 5 Nov frames; a gate only if the data gives one (decision 5); verified parked (**D3**) before it drives. **P3** second lighting setup.

**Gate:** D4a, D4b, D6 (scoped) PASS parked; `routine-stop-sign` PASS on ≥ 5 sign and ≥ 5 no-sign trials; D3 PASS under both lighting setups.

### Phase 4: the campaign (2 course sessions on different days)

1. **T2** D1 criteria in the course routines (decision 1).
2. **P5** 5-minute `diag-run` + `pi-load` (R4 memory). **P6** watchdog hardware test (R5).
3. Course run 1: `make routine-figure-eight` ≥ 10 min, then `approach-log`, `nav-report`, `fig8-report`, `light-report`. Judge P1, P2, D1, D2, D5 from it.
4. Course run 2, another day (**P1** for D5's second run), same reports.
5. `requirements.md` status columns updated from the two runs; "Changes to pass rules" for anything that moved.

**Gate:** every row of `requirements.md` reads *Met* or carries an agreed scope note (D6 dark objects), with the run folder and date beside it.

### What is deliberately not in this plan

- Dead reckoning during a lane hold, the ground homography for lane offset in cm, and process-level parallelism: each is in a doc's "Open items" and none is needed to meet a current requirement.
- A per-intersection light association model (F9's heavy form): only if F9's measurement shows no simple gate.
- New analysis tools: every measurement the plan needs has one already (§3.2).
