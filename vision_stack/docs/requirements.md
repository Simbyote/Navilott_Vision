# Requirements and Verification

> What the vision system must do, the rule each requirement is judged by, how it's checked, and where it stands.

Each requirement has a **pass rule** and one way to check it. A pass rule names three things: the **threshold**, the **statistic** (mean, median, p95, k of n) and the **condition** it applies under (bench or driving, which range, which objects). The rule is fixed before the test is run. If it changes, the change is recorded under [Changes to pass rules](#changes-to-pass-rules) along with the reason, never quietly.

A requirement is **Verified** only when its check has passed on the current hardware: the IMX290 camera at 480×270, remounted upright, on the Pi Zero 2 W. Results from the IMX219 at 480×360 are kept as history but no longer count.

**Verdicts:**
- **Met:** the check passed under the stated condition.
- **Not met:** the check ran and missed. The status says by how much.
- **Not verified:** the check hasn't run, or ran only in part.

**Methods (IADT):**
- **T:** test, measured against an external reference (tape, a ruler, the light state that was set).
- **A:** analysis, scored against labelled frames.
- **D:** demonstration, the robot visibly does it on the course.
- **I:** inspection, by design or review.

**Proposed** values were filled in on 2026-11-02 where the PSR left them open. They follow from navigation's needs and the routines' existing settings. The team must agree them before the tests that use them are run; see [Open decisions](#open-decisions).

**Counting:** a rate is reported as k / n with its 95% interval, and n counts separate placements, approaches or trials, not frames. Consecutive frames of one scene aren't independent.

---

## Performance

| ID | Requirement | Pass rule | Method | How it's checked | Status |
| --- | --- | --- | --- | --- | --- |
| P1 | Process at least **20 frames per second**, sustained | **While driving**, over a run of at least 5 minutes: effective FPS (frames ÷ wall time in `summary.txt`) ≥ 20.0, and ≤ 5% of frames over the 50 ms budget (P2 + P3 > 50 ms) | T | Any recorded course run (`make routine-figure-eight`, `make routine-course-run`): the run's `summary.txt`. The bench routine `make routine-frame-budget` (3 × 60 s, motors off) is a diagnostic. | **Not met** (2026-10-28, 10-min figure-eight): 19.3 FPS effective, 324 of 11,629 frames (2.8%) over budget; Phase 2 averaged 38.9 ms while driving. On the bench: 20.0 FPS. |
| P2 | Deliver each navigation update within **50.0 ms** of the frame being captured | **While driving:** p95 of P2 + P3 time (the `total` row in `summary.txt`) ≤ 50.0 ms. The clock starts when the frame reaches the application, after exposure and the ISP; see note. | T | The same run's `summary.txt`. Also report "command latency" (frame in → motor write) next to it. | **Not met** (2026-10-28, driving): P2 + P3 p95 51.8 ms; frame → motors p95 52.9 ms. On the bench: 40.7 ms frame → motors. |
| P3 | Report lateral position in the lane to within **±2 cm** | Parked at ruler-measured offsets of 0, ±2 and ±4 cm from lane centre, on 2 straights (10 placements): the mean reported `lane_offset_cm` within ±2.0 cm of the ruler at every placement. The 0 cm look also sets the scale (14 cm ÷ lane width in px); its own error tests the centre (`robot_x_px`), and the ±2 and ±4 cm looks test the scale. | T | `make routine-lane-offset`; see [Procedure for P3](#procedure-for-p3-lane-offset-accuracy) | Not verified |
| P4 | Run entirely on board, with no network connection | Nothing in the pipeline opens a network connection | I | By design | Met by design |

On P2: the pipeline stamps a frame when it reaches the application, after exposure and ISP processing. Time from exposure to that stamp is a few frame periods at most but hasn't been measured. If P2 means "light to decision", it needs a separate measurement, such as filming an LED next to the robot's display.

On P1: the camera is set to 20 FPS, so effective FPS can never read above 20. It shows a shortfall, but not how much spare time there is. The spare time per frame is 50 ms minus P2 + P3. Report that next to P1.

---

## Detection

All distances are measured from the **robot's front edge** to the **near edge** of the line, or the base of the object. This is the reference `stop_line_table.json` uses. The stop line is out of the camera's view closer than about **8 cm**.

| ID | Requirement | Pass rule | Method | How it's checked | Status |
| --- | --- | --- | --- | --- | --- |
| D1 | Detect the lane boundaries of the robot's own lane | **Proposed.** On straight lane-keeping segments while driving: lane status `vision` in ≥ 90% of frames; no `stale` stretch longer than 1.0 s. Parked, centred: lane-mode flicker ≤ 5% of 300 frames. Correctness is P3, not D1. | T, A | Lane-keeping stretches of a recorded course run (`summary.txt` lane-status block, split by rule with `make timeline`); parked: `make phase3 ARGS="--limit 300"` then `make stability` | Not verified against the target. 2026-10-28, whole 10-min run including intersections and turns: `vision` 75.1%, `hold` 8.4% (longest 7 frames), `stale` 16.5% (longest 59 frames, steered by gyro heading). |
| D2 | Detect a stop sign at the expected range | **Proposed range.** Parked at 0, 10 and 20 cm before the sign's stop line: the sign read in ≥ 50% of a look's frames and voted present. Sign taken away: **0** wrong reads at any gap, including with the red traffic lamp lit and red objects in view. On the course: every sign stopped for (10 / 10 trials), and **at most 1 false stop per 10 no-sign trials, or per 10 minutes driven** (agreed 2026-11-02). | T, D | `make routine-detect-range ARGS="--set target=sign"` (plain, then with the red lamp lit for the "absent" looks); `make routine-stop-sign` (10 sign and 10 no-sign trials); false stops counted over a recorded course run | **Not met** (2026-10-28, figure-eight): 6 / 6 real signs stopped for, but 3 false stops in 10 min against at most 1 (red traffic lamp ×2, red bags ×1). Range not yet measured. History: shape gates fitted to the course's sign (2026-10-20). |
| D3 | Classify a traffic light as red, yellow or green | **Proposed range.** Parked at 0, 10 and 20 cm before the light's stop line, each state set in turn (red, yellow, green, off): read right. **0** wrong-colour reads at any gap, out to 45 cm. **Yellow voted as stop (read as red) counts as right**: navigation stops for both (agreed 2026-11-02); yellow voted go is a wrong read. Repeated under 2 lighting setups. | T | `make routine-detect-range ARGS="--set target=light"` | **Not met as scored** (2026-10-28, labelled frames from the run's log, 4,375 frames): read right per colour, red 64%, yellow 3% (2 / 59), green 68%, off 95%. At stop lines: 1 red stop and 5 greens, with no yellow at a line. Range not yet measured. History: 196 / 196 frames at the stop read right (2026-10-06, earlier build). |
| D4 | Detect the stop line at an intersection, and how far ahead it is | **D4a, detection:** seen in ≥ 95% of frames at every gap 8–20 cm (2 cm steps), square and at ±10°; **0** stop-line frames in 300 frames of plain lane with a dashed centre line. **D4b, distance:** at new placements (not the calibration's own), the median `stop_line_distance_cm` within ±0.5 cm of the tape at ≤ 10 cm and ±1.0 cm beyond, with a frame-to-frame SD < 0.5 cm. Report gaps beyond the calibration's fit range (8–16 cm) separately. | T | D4a: `make routine-detect-range ARGS="--set target=line --set gaps=8,10,12,14,16,18,20 --set angles=-10,0,10 --set frames=100"`, plus `make phase3 ARGS="--limit 300"` on plain lane. D4b: `make phase3 ARGS="--limit 100"` per tape mark (9, 11, 13, 15, 18 cm) | Not verified. Needs a stop-line calibration that matches the current lens: the committed `stop_line_table.json` (2026-10-22) is rejected at load because it was fitted with an older lens calibration (2026-10-26). |
| D5 | Detect intersections | Every crossing judged PASS by `intersection_linker`, over **≥ 2 separate runs**. Report how many physical intersections the course has: laps repeat the same ones, so they aren't independent trials. | D | `make intersection MANEUVER=all`; `make routine-figure-eight` (8 crossings a lap), then `make fig8-report` | Demonstrated in one run (2026-10-28): 44 / 44 crossings in order during the 10-min figure-eight. A second run is still needed. |
| D6 | Detect an obstacle in the robot's path, how far ahead its base is, and stop for it within about 10 cm until it's removed | **Detection:** for each object class, at gaps of 10–25 cm, seen in ≥ 90% of frames and the median distance within ±1.5 cm of the tape. Path clear: < 2% of frames fire. **Stop:** stops 5–15 cm before the object on every trial, drives on once it's lifted, and never stops on a clear trial. **Scope (agreed 2026-11-02):** objects that contrast with the mat, tested with a light box and the course's bottle. **Dark objects are a known limitation**: a dark object (shoe) is still measured the same way and reported, but not judged. | T, D | `make routine-obstacle-check ARGS="--set object=<name> --set tolerance_cm=1.5"` per object; `make routine-obstacle-stop` (≥ 5 object and 3 clear trials per object) | Not verified on the bench. 2026-10-28, figure-eight: a coloured bottle in the lane was found and the robot held until it was lifted (1 event). A person's dark shoes were missed, 0 / 63 frames (the known limitation). 2 false frames in 3,639 with nothing there, 0 after the vote. Bench distances not yet measured. |

D5 was first planned as a scene state machine, a separate detector for "the robot is in an intersection". It was built a simpler way instead: every intersection on the course has a stop line at the robot's entry, so the stop line (D4) marks it, and the route says which way to go. An intersection with no stop line wouldn't be seen; on this course there isn't one.

**Why the ranges are 20 cm (D2, D3, D4a).** Navigation brakes for the stop line and has to know the light or sign before it reaches the line. 20 cm is `detect-range`'s `need_cm`. At base speed it gives about a second of warning, many times the 3-frame vote. Readings farther out are recorded to map the range, but not judged.

**Why the obstacle range is 25 cm (D6).** Stopping within about 10 cm of an object needs the object seen and measured from farther out than the stopping distance. 25 cm is `obstacle-check`'s `need_cm`. The routine's own defaults (50% of frames, ±3 cm) are looser than D6, so the D6 values are passed with `--set`.

**Scoring the detectors on recorded frames (labelled replay).** The routines above judge one condition live. To measure a detector across many conditions, and again after every tuning change on the same frames:
1. Record a run with frames (any linker).
2. Label what was really in view with `make label RUN=...`. It writes `labels.csv`: ranges of frames with the sign, the light, the stop line's and an obstacle's tape distance, the lane offset, the range to the sign or light, and a condition tag.
3. Run `make eval RUN=...`. It replays every frame through the robot's own `Pipeline` and scores each detector frame by frame and as voted, written to `eval.json` and `eval.png`:
   - recall and precision with 95% intervals (Wilson);
   - false alarms;
   - the light's confusion matrix;
   - distance errors against the tape;
   - all broken down by range and condition.
4. Pass `BASE=` an earlier `eval.json` to print the change.

A detect-range, obstacle-check or lane-offset routine folder labels itself from its saved looks' names.

This is method **A**, supporting evidence. It doesn't replace the T and D checks in the tables:
- Report results per class, never overall accuracy alone.
- Say how the labelled frames were chosen.
- Say who labelled them, and that they couldn't see the detector's output.

---

## Robustness

These are internal targets from the design, not PSR requirements. They're here so each has a check.

| ID | Target | Pass rule | Method | How it's checked | Status |
| --- | --- | --- | --- | --- | --- |
| R1 | A single dropped frame never stops the robot; a dead camera is reported within about 1 s | As the test asserts | T | `test_capture` (software) | Verified in software |
| R2 | A lost lane is held for at most about 350 ms, then reported as unusable | As the test asserts | T | `test_estimation` (software) | Verified in software. Seen on the course (2026-10-28): longest hold 7 frames (350 ms). |
| R3 | No single frame can flip the traffic or stop-sign state | As the test asserts | T | `test_estimation` (software) | Verified in software |
| R4 | CPU under 70% and memory under 400 MB with navigation running | **While driving:** process CPU (mean over the run) < 70% and the **busiest core's** p95 < 90%; peak RSS < 400 MB | T | `make diag-run`, then `make pi-load`. `make routine-frame-budget` is the bench diagnostic. | Not verified while driving. On the bench (frame-budget, motors off): 41% process CPU, 157 MB peak. Per-core not yet reported. |
| R5 | The motors stop if control stops arriving | Brake within `MOTOR_WATCHDOG_S` (0.5 s) of the last command | T | `test_drive` (software): the watchdog in `drive.py`. Hardware: `pytest --hardware -k watchdog` with the wheels up | Verified in software; the hardware test is written, not yet recorded |

---

## Procedure for P3 (lane offset accuracy)

`make routine-lane-offset` prompts through these steps and judges them.

1. **Measure the ground scale.** Park the robot centred in a straight lane and run `phase3_linker --camera --limit 100`. Note `L` and `R` in the status line. The ground scale is 14 cm ÷ (R − L) px, using the lane width measured the same way: centre of one line to centre of the other (see `course.md`).
2. **Record the noise floor.** Still parked and centred, run `--limit 300` and note the offset `std` in `summary.txt`.
3. **Measure at known offsets.** Shift the robot 2 cm and then 4 cm right of centre, then the same to the left. Measure each offset with a ruler, from the lane centre to the robot's centreline. At each position run `phase3_linker --camera --cm-per-px <scale> --limit 100` and note the mean `lane_offset_cm` in `p3.csv`.
4. **Repeat on a second straight.** That gives 10 judged placements in total.
5. **Pass:** at every placement, the mean reported offset is within 2 cm of the ruler.

Record the results in the status column above, with the date and commit.

---

## Open decisions

The proposed values below are what the plans and test sheets use. Agree them with the team, or change them, **before** the tests run. Then remove "Proposed" from the row. Rows marked **Agreed** are settled.

| # | Requirement | Proposed | Why | Decide |
| --- | --- | --- | --- | --- |
| 1 | D1 target | `vision` ≥ 90% on straight lane-keeping segments; `stale` ≤ 1.0 s; flicker ≤ 5% parked | Lane keeping steers from the offset only when the lane is on `vision`; `stale` hands steering to the gyro | The share, and whether turns and intersections are excluded |
| 2 | D2 / D3 range | Read right at every gap ≤ 20 cm; 0 wrong reads out to 45 cm | Navigation must know before the stop line; 20 cm is about a second at base speed | The range, from the PSR if it gives one |
| 3 | D2 false stops | **Agreed 2026-11-02:** at most 1 per 10 no-sign trials, or per 10 minutes driven; every real sign still stopped for | A stop with no sign is a traffic violation, but a rare one doesn't end the run | Done |
| 4 | D3 yellow | **Agreed 2026-11-02:** yellow voted stop (read as red) passes; voted go fails | Yellow stops the robot like red does (`navigation/traffic_light.py`), so the behaviour is the same | Done |
| 5 | D4 tolerances | ±0.5 cm at ≤ 10 cm, ±1.0 cm beyond | `stop-distance` aims to stop 2–6 cm before the line, a 4 cm window | Confirm against navigation's stopping spread |
| 6 | D6 object classes | **Agreed 2026-11-02:** scoped to objects that contrast with the mat; dark objects measured and reported as a known limitation | The 28 Oct run shows the detector is blind to dark objects; fixing it is future work | Done |
| 7 | R4 per core | Busiest core p95 < 90% | The main thread carries Phase 2; one saturated core explains a latency miss even when the average is low | The threshold |

---

## Changes to pass rules

| Date | Requirement | Change | Reason |
| --- | --- | --- | --- |
| 2026-11-02 | P1, P2, R4 | Judged **while driving**. The bench routine became a diagnostic. | The requirements describe a driving robot; bench numbers were being read as verdicts. |
| 2026-11-02 | P3 | A second straight added | More independent placements. (The 0 cm look sets only the scale; its offset comes from `robot_x_px`, so it is judged like the others.) |
| 2026-11-02 | D4 | Reference changed to the robot's front edge; gaps now 8–20 cm instead of 0–20 cm; split into D4a / D4b; new placements required | Matches `stop_line_table.json`. Since the camera was remounted upright, the line leaves the view below about 8 cm. |
| 2026-11-02 | D1, D2, D3, D6 | TBD values filled with **proposed** rules | See Open decisions. |
| 2026-11-02 | D2 | False stops: at most 1 per 10 no-sign trials or 10 minutes driven, instead of 0 | Agreed with the team: real signs must always be stopped for; a rare false stop is tolerated |
| 2026-11-02 | D3 | Yellow read as red counts as right | Agreed with the team: navigation stops for both, so the robot's behaviour is correct |
| 2026-11-02 | D6 | Scoped to objects that contrast with the mat; dark objects a known limitation, measured but not judged | Agreed with the team, before the bench tests |

---

## Status summary

| | Met | Not met | Not verified |
| --- | --- | --- | --- |
| Performance | P4 | P1, P2 | P3 |
| Detection | | D2 (false stops), D3 (as scored) | D1, D4, D5 (one run so far), D6 (bench) |
| Robustness | R1, R2, R3, R5 (in software) | | R4 (while driving) |
