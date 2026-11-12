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

## Requirement statements (FDR form)

Each presented requirement in the five parts the FDR asks for: what is measured, the target, the tolerance or
threshold, the conditions it must hold over, and how it is verified. They restate the tables below; where the two
differ, the tables (and the change log) win.

| ID | What is measured | Target | Tolerance / threshold | Conditions | Verified by |
| --- | --- | --- | --- | --- | --- |
| P1 | Frames processed per second | 20 FPS | ≥ 20.0 effective; ≤ 5% of frames over 50 ms | while driving, a run of ≥ 5 min | Test: course run, `summary.txt` |
| P2 | Time from a frame's arrival to its packet (capture + Phase 2 + Phase 3) | 50 ms | 95th percentile ≤ 50 ms | while driving, every frame of the run | Test: course run, `summary.txt` "total" row |
| P3 | Reported lane offset vs a ruler | 0 cm error | ±2 cm at every placement, **and** a best-fit slope of 0.8–1.2 against the ruler (revised 2026-11-05) | robot 0 to ±4 cm off centre (1 cm steps), parked on a straight, both lines in view, lab lighting | Test: `make routine-lane-offset`, ≥ 2 straights |
| D1 | Share of frames with the lane found | 90% | lane status `vision` in ≥ 90% of frames (agreed 2026-11-12) | while driving, the frames lane keeping decides (no other rule or maneuver) | Test: course run (`make routine-figure-eight` or `routine-course-run`, which judge it) |
| D2 | Stop sign confirmed (voted over 3 frames) before the line / confirmed with no sign there | every sign intersection; none without a sign | confirmed before the stop line at **every** sign intersection; **≤ 1** false confirmed sign per 10 min driven (revised 2026-11-05) | while driving the course; parked runs characterise the range (in view from ~20 cm, front-lit) | Test: course run (`make routine-figure-eight`), cut per intersection by `make approach-log` |
| D3 | Light state decided at the line | every state right within 20 cm | within 20 cm: red or yellow decided stop **or** caution, green or off decided go; at **any** distance, never go for a lit red or yellow (revised 2026-11-05) | parked at 0–45 cm, 2 lighting setups | Test: `make routine-detect-range ARGS="--set target=light"` |
| D4a | Stop line seen | every pose in view | ≥ 95% of frames, pooled over all looks at 8–16 cm, square and ±10° (revised 2026-11-05); 0 line frames in 300 frames of plain lane | 8–16 cm (in view) | Test: `make routine-detect-range ARGS="--set target=line"` |
| D4b | Reported stop-line distance vs tape | 0 cm error | **±2 cm** (revised 2026-11-05); frame-to-frame SD < 0.5 cm where a look logs it | 8–17 cm from the robot's front edge, new placements | Test: same runs, `make line-replay` |
| D5 | Intersections crossed as routed | every crossing | 100% judged PASS over ≥ 2 separate runs | the course, as driven | Demonstration: `make routine-figure-eight` |
| D6 | Obstacle seen and its distance | every placement | ≥ 90% of frames and median within **±2.5 cm** at **15–25 cm** (revised 2026-11-05); < 2% false frames with the lane clear | objects that contrast with the mat (dark objects: known limitation) | Test: `make routine-obstacle-check` per object |
| R1–R3, R5 | Recovery from a dropped frame, a lost lane, a single bad frame; motors stop without commands | as specified | as each test asserts | software, every commit | Test: the pytest suite |
| R4 | CPU and memory | < 70% CPU, < 400 MB | process mean; busiest core p95 < 90% | while driving | Test: `make diag-run`, `make pi-load` |

**To state at FDR (changes from the PSR):** compare each target above with the PSR's. Where one was relaxed or
re-scoped, give the reason on the summary slide before anyone asks, for example: the latency target if the PSR
said 33.3 ms (30 FPS); the stop-sign and stop-line ranges (out of the camera's view closer than ~15–20 cm and
past ~17 cm); dark obstacles out of scope.

---

## Performance

| ID | Requirement | Pass rule | Method | How it's checked | Status |
| --- | --- | --- | --- | --- | --- |
| P1 | Process at least **20 frames per second**, sustained | **While driving**, over a run of at least 5 minutes: effective FPS (frames ÷ wall time in `summary.txt`) ≥ 20.0, and ≤ 5% of frames over the 50 ms budget (Phase 2 + Phase 3 over 50 ms) | T | Any recorded course run (`make routine-figure-eight`, `make routine-course-run`): the run's `summary.txt`. The bench routine `make routine-frame-budget` (3 × 60 s, motors off) is a diagnostic. **Diagnostic, capture throughput:** `make capture-anatomy` (camera alone) gives the frame period, frames lost and frame age, which tells a shortfall in capture from one in processing. | **Not met** (2026-11-04, 9.3-min course run, 10,664 frames; `routine_figure-eight_20261104_165132`): 19.2 FPS effective. Earlier (2026-10-28, 10-min figure-eight): 19.3 FPS, 324 of 11,629 frames (2.8%) over budget; Phase 2 averaged 38.9 ms while driving. On the bench: 20.0 FPS. Capture alone (2026-10-28, `capture-anatomy`, 200 frames): a frame every 49.996 ms, none lost, no CPU time on the capture path: the shortfall is processing, not the camera. |
| P2 | Deliver each navigation update within **50.0 ms** of the frame being captured | **While driving:** p95 of P2 + P3 time (the `total` row in `summary.txt`) ≤ 50.0 ms. The clock starts when the frame reaches the application, after exposure and the ISP; see note. | T | The same run's `summary.txt`. Also report "command latency" (frame in → motor write) next to it. | **Not met** (2026-11-04, 9.3-min course run): capture + Phase 2 + Phase 3 p95 52.2 ms (median 43.0); command latency, frame in → motors, p95 54.5 ms. Stage medians: Phase 2 40.1 ms, of it geometry 26.3 and preprocessing 12.0; capture 3.6; Phase 3 0.6. Earlier (2026-10-28, driving): Phase 2 + Phase 3 p95 51.8 ms; frame → motors p95 52.9 ms. On the bench: 40.7 ms frame → motors. |
| P3 | Report lateral position in the lane to within **±2 cm** | Parked at ruler-measured offsets of 0, ±2 and ±4 cm from lane centre, on 2 straights: the mean reported `lane_offset_cm` within ±2.0 cm of the ruler at every placement, **and** (revised 2026-11-05) the best-fit slope of reported against ruler within 0.8–1.2: lane keeping steers on this number, so a compressed scale weakens every correction even when each placement is inside ±2 cm. The 0 cm look sets the scale (14 cm ÷ lane width in px) and tests the centre (`robot_x_px`). | T | `make routine-lane-offset`; see [Procedure for P3](#procedure-for-p3-lane-offset-accuracy) | **Not met** (2026-11-03, 4 runs, 32 placements): 19 / 32 within ±2 cm; slope 0.63 (best fit 0.63 × true − 1.15 cm), under 0.8. One run on the far side of the lane, one line in view, read up to 4 cm wrong. Not yet re-run after `robot_x_px` = 224 (2026-11-05). |
| P4 | Run entirely on board, with no network connection | Nothing in the pipeline opens a network connection | I | By design | Met by design |

On P2: the pipeline stamps a frame when it reaches the application, after exposure and ISP processing. Time from the sensor's timestamp to that stamp is 31.6 ms median (31.8 p95, 34.6 max; `make capture-anatomy`, 2026-10-28, camera alone): readout, Unicam and the ISP, plus 0.5 ms through GStreamer. A frame is about 32 ms old when P2's clock starts, so light to decision is about 75 ms at the median. An exact light-to-decision figure, from the moment light reaches the sensor, still needs a separate measurement, such as filming an LED next to the robot's display.

On P1: the camera is set to 20 FPS, so effective FPS can never read above 20. It shows a shortfall, but not how much spare time there is. The spare time per frame is 50 ms minus P2 + P3. Report that next to P1.

---

## Detection

All distances are measured from the **robot's front edge** to the **near edge** of the line, or the base of the object. This is the reference `stop_line_table.json` uses. The stop line is out of the camera's view closer than about **8 cm**.

| ID | Requirement | Pass rule | Method | How it's checked | Status |
| --- | --- | --- | --- | --- | --- |
| D1 | Detect the lane boundaries of the robot's own lane | **Agreed 2026-11-12.** While driving, lane status `vision` in **≥ 90%** of the frames **lane keeping decides** (`nav.csv`'s `rule` = `lane_keeping`): frames where another rule acts (an intersection's maneuver, a hold at a sign or light, an obstacle, the end of course) aren't counted. Correctness is P3, not D1. | T | `make routine-figure-eight` / `routine-course-run` judge it ("lane on vision while lane keeping (D1)", `lane_vision_pct`); for an older run, the same count over its `nav.csv` | **Met on the two recorded course runs**, counted from their logs after the rule was agreed: 28 Oct (10 min), **98.6%** of 5,275 lane-keeping frames (`hold` 1.4, `stale` 0); 4 Nov (9.3 min), **96.4%** of 2,874 (`hold` 3.6, `stale` 0). To confirm on a run under the rule. *As first proposed:* ≥ 90% on straight lane-keeping segments, no `stale` stretch over 1 s, and ≤ 5% flicker parked; the last two not adopted. |
| D2 | Detect a stop sign at the expected range | **Revised 2026-11-05: judged while driving, per intersection.** Navigation acts on a sign confirmed over 3 frames and remembers it 5 s (`SIGN_MEMORY_MS`), so the sign has to be confirmed somewhere on the approach, not in every parked frame. On the course: the sign confirmed before the stop line at **every** sign intersection; **at most 1** false confirmed sign per 10 minutes driven (the false-stop allowance agreed 2026-11-02). Parked runs now characterise the range and the lighting it needs; they are reported, not judged. *Originally:* parked at 0, 10 and 20 cm, read in ≥ 50% of a look's frames and voted present; 0 wrong reads with the sign away. | T, D | A course run cut per intersection (`make routine-figure-eight`, then `make approach-log ARGS="--sign-at <lap positions>"`); parked: `make routine-detect-range ARGS="--set target=sign"` | **Not met** (2026-11-04, figure-eight, 9.3 min, 37 intersections): confirmed at 4 of 5 sign intersections (first seen 25–60 cm out; lap 1 intersection 6 missed); falsely confirmed at 3 of 32 without a sign, 3 in 9.3 min against at most 1 per 10, each from 1–3 frames beside a red traffic light. Parked (2026-11-02, characterisation): in view from ~20 cm, front-lit 15 / 16, backlit 0 / 8; out of view closer than ~15 cm; 0 / 96 false votes with the sign away. *As originally written:* not met. |
| D3 | Classify a traffic light as red, yellow or green | **Revised 2026-11-05.** Navigation stops for both red and caution (`traffic_light.py`), so red and yellow are judged by whether the decision stops the robot, and the one dangerous error is go for a lit red or yellow. Parked at 0–20 cm, each state set in turn: red or yellow decided stop or caution, green or off decided go. At **any** distance out to 45 cm: never go for a lit red or yellow. Repeated under 2 lighting setups. *Originally:* every state read as its own colour, and 0 wrong colours out to 45 cm (yellow voted stop already counted as right, agreed 2026-11-02). | T | `make routine-detect-range ARGS="--set target=light"` | **Met for one lighting setup** (2026-11-04, 3 runs, lab lighting, 0–45 cm, 80 looks): within 20 cm, 60 / 60; out to 45 cm, no go for a red or yellow (the one miscolour, red read as yellow at 45 cm, still stops). The second lighting setup is still owed. Driving (2026-11-04): 6 of 18 unlit lights were decided red or yellow, each next to a lit light in view: a false wait, not a go-on-red; see the change log. *As originally written:* not met (the one wrong colour at 45 cm). |
| D4 | Detect the stop line at an intersection, and how far ahead it is | **D4a, detection (revised 2026-11-05):** seen in ≥ 95% of frames pooled over all looks at 8–16 cm (2 cm steps), square and at ±10°: the tracker needs the line while it is in view, and past ~17 cm it is outside the lane view; **0** stop-line frames in 300 frames of plain lane. **D4b, distance (revised 2026-11-05):** at new placements, the median `stop_line_distance_cm` within **±2 cm** of the tape, frame-to-frame SD < 0.5 cm where a look logs it. The distance sets where the robot stops: it stops with ~9 cm of line ahead (`REACH_CM`, chosen so the light stays in view), and ±2 cm keeps that stop clear of the line and the light in view. *Originally:* D4a ≥ 95% at every gap 8–20 cm and angle; D4b ±0.5 cm to 10 cm, ±1.0 cm beyond. | T | D4a: `make routine-detect-range ARGS="--set target=line --set gaps=8,10,12,14,16 --set angles=-10,0,10 --set frames=100"`, plus `make phase3 ARGS="--limit 300"` on plain lane. D4b: the same looks, `make line-replay` | 2026-11-02, 3 runs × 7 gaps × 5 angles. **D4a met** (revised): 98% of frames pooled over the 75 looks at 8–16 cm (73 looks at ≥ 95% each; two at −10° near the bottom of the view short); the plain-lane check is still owed. **D4b not met:** 34 / 57 within ±2 cm; it reads +1.7 cm long (median), a steady offset to recalibrate. Replayed from one frame a look (no SD). *As originally written:* both not met. |
| D5 | Detect intersections | Every crossing judged PASS by `intersection_linker`, over **≥ 2 separate runs**. Report how many physical intersections the course has: laps repeat the same ones, so they aren't independent trials. | D | `make intersection MANEUVER=all`; `make routine-figure-eight` (14 crossings a lap since 2026-11-02, 8 before), then `make fig8-report` | Demonstrated in one run (2026-10-28): 44 / 44 crossings in order during the 10-min figure-eight. A second run is still needed. |
| D6 | Detect an obstacle in the robot's path, how far ahead its base is, and stop for it within about 10 cm until it's removed | **Detection (revised 2026-11-05):** for each object class, at gaps of **15–25 cm**, seen in ≥ 90% of frames and the median distance within **±2.5 cm** of the tape. Navigation stops 5–15 cm before the object, so it needs the object from 15 cm out (at 10 cm it has already stopped), and a 2.5 cm error keeps a stop aimed at 10 cm inside that band. Path clear: < 2% of frames fire. **Stop:** stops 5–15 cm before the object on every trial, drives on once it's lifted, and never stops on a clear trial. **Scope (agreed 2026-11-02):** objects that contrast with the mat. **Dark objects are a known limitation**: measured and reported, not judged. *Originally:* 10–25 cm, ±1.5 cm. | T, D | `make routine-obstacle-check ARGS="--set object=<name>"` per object; `make routine-obstacle-stop` (≥ 5 object and 3 clear trials per object) | **Met for the yellow cube and the solder box** (2026-11-03, 3 runs each): 9 / 9 placements at 15–25 cm, every one seen in 100% of frames and within ±2.5 cm (largest error 2.1 cm). **Not met for the blue can:** 0 / 9 (seen only at 10 cm; dark and narrow, so it may belong under the dark-object limitation, to decide). Path clear: 27 / 27 trials with no false frames. Stop behaviour not yet tested. *As originally written:* not met (10 cm out of view; 3 errors over 1.5 cm). |

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
| R4 | CPU under 70% and memory under 400 MB with navigation running | **While driving:** process CPU (mean over the run) < 70% and the **busiest core's** p95 < 90%; peak RSS < 400 MB | T | an unlogged course run (`make run ROUTE=route_figure_eight.json`, recorded with `make monitor-attach`; `phase1_robot_procedure.md`, step 8), then `make pi-load`, whose **R4** line judges all four numbers (added 2026-11-12). `make routine-frame-budget` is the bench diagnostic. | **Verified while driving** (2026-10-28, `make diag-run` on the course, `src.main --route route.json`, 133 s, diag_20261028_131532): CPU 40% of the Pi (mean, every process; the run itself 36%), busiest core p95 74% (`pi-load`'s R4 line computes 69.9% from that recording's `cores.csv`), peak RSS 148 MB; no throttling, 59 °C max. The run was too short (2.2 min) to judge memory growth, which needs 5. On the bench (frame-budget, motors off): 41% CPU, 157 MB peak. |
| R5 | The motors stop if control stops arriving | Brake within `MOTOR_WATCHDOG_S` (0.5 s) of the last command | T | `test_drive` (software): the watchdog in `drive.py`. Hardware: `make test-hw ARGS="src/tests/test_drive.py -k watchdog"` with the wheels up (`phase1_robot_procedure.md`, step 9) | Verified in software; the hardware test is written, not yet recorded |

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
| 1 | D1 target | **Agreed 2026-11-12:** `vision` ≥ 90% of the frames lane keeping decides | Lane keeping steers from the offset only on `vision`; when another rule acts (a maneuver, a hold, an obstacle) the lane isn't what the robot steers by | Done |
| 2 | D2 / D3 range | Read right at every gap ≤ 20 cm; 0 wrong reads out to 45 cm | Navigation must know before the stop line; 20 cm is about a second at base speed | The range, from the PSR if it gives one |
| 3 | D2 false stops | **Agreed 2026-11-02:** at most 1 per 10 no-sign trials, or per 10 minutes driven; every real sign still stopped for | A stop with no sign is a traffic violation, but a rare one doesn't end the run | Done |
| 4 | D3 yellow | **Agreed 2026-11-02:** yellow voted stop (read as red) passes; voted go fails | Yellow stops the robot like red does (`navigation/traffic_light.py`), so the behaviour is the same | Done |
| 5 | D4 tolerances | ±0.5 cm at ≤ 10 cm, ±1.0 cm beyond | `stop-distance` aims to stop 2–6 cm before the line, a 4 cm window | Confirm against navigation's stopping spread |
| 6 | D6 object classes | **Agreed 2026-11-02:** scoped to objects that contrast with the mat; dark objects measured and reported as a known limitation | The 28 Oct run shows the detector is blind to dark objects; fixing it is future work | Done |
| 7 | R4 per core | Busiest core p95 < 90% | The main thread carries Phase 2; one saturated core explains a latency miss even when the average is low | The threshold |
| 8 | D4b: the stop-line table | **Decided 2026-11-12:** re-fit it on the track (`make calib-stopline`) and re-evaluate there, rather than shift its `C` by the 2 Nov median | The obstacle's cm come from the same curve, so D6 is re-checked after the re-fit too (`make routine-obstacle-check`) | Done; the re-fit is the team's next track session |

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
| 2026-11-04 | D3 | Considered: parked clips of the light cycling, labelled frame by frame, scored by `make light-clips`. **Not adopted**: TP-03 stays one state at a time (`detect-range`, target=light), which labels itself; the clip tool stays available for response-time checks | Labelling every clip by hand was not worth it for the FDR; the one-state routine already covers each colour at each gap |
| 2026-11-05 | D2, D3, D4, D6, P3 | **Revised after testing**, to what Navigation relies on (agreed with the instructor: targets that don't reflect how the system uses an output get revisited). D2 judged per intersection while driving (confirmed before the line at every sign; ≤ 1 false per 10 min); parked sign runs become characterisation. D3: red and yellow judged by whether the decision stops the robot; never go for a lit red or yellow at any distance. D4a: ≥ 95% of frames pooled over 8–16 cm. D4b: ±2 cm. D6: 15–25 cm, ±2.5 cm. P3: adds a slope rule, 0.8–1.2 | The pipeline was built to be forgiving: each detection is confirmed over 3 frames, the sign is remembered 5 s, and Navigation stops for caution as for red, ~9 cm before the line, and 5–15 cm before an obstacle. Single-frame parked targets set before the design was finished judged precision Navigation never uses. Each new limit comes from those constants, not from the data. Results are reported against both the original and the revised rule. Not revised: P1, P2 (nothing downstream sets a looser budget) |
| 2026-11-05 | D3 (driving) | Recorded, not yet a rule: on the course, 6 of 18 unlit lights were decided red or yellow, each next to a lit light in view | The parked test has one light in view; the course has several. Tying a light to its own intersection is open work |
| 2026-11-12 | P1, P2, D2, D3 | **Date corrected, no rule changed:** the 9.3-min course run these rows quote is `routine_figure-eight_20261104_165132`, recorded 2026-11-04 16:51, not 2026-11-05 | The routine folder's own time stamp; its frames (10,664) and command-latency p95 (54.5 ms) match the rows |
| 2026-11-12 | D1 | **Target agreed:** `vision` ≥ 90% of the frames lane keeping decides; every other rule's frames left out. The proposed no-stale-over-1-s and parked-flicker clauses not adopted | Agreed with the team (first stated as "outside intersection crossing stages", narrowed the same day to lane keeping's own frames). Judged by the course routines from this date; the two earlier course runs counted from their logs |

---

## Status summary

Each row's own verdict, as defined at the top (Met, Not met, Not verified), on the current hardware. *Checked* is not *met*: every requirement has had its check run except where marked Not verified. Updated 2026-11-12 from the rows above; no result or rule changed.

| | Met | Not met | Not verified, or in part |
| --- | --- | --- | --- |
| Performance | P4 (by design) | P1 (19.2 FPS), P2 (p95 52.2 ms), P3 (slope 0.63) | |
| Detection | D1 (from two runs' logs, 98.6% / 96.4%), D4a (pooled 98%) | D2 (3 false stops in 9.3 min), D4b (+1.7 cm) | D3 (one of two lighting setups), D4a's plain-lane check, D5 (one run of two), D6 (met for the cube and box; the blue can not, its scope to decide) |
| Robustness | R1, R2, R3 (software), R4 (driving, 2.2 min) | | R4's memory trend (needs ≥ 5 min), R5 on hardware (written, not recorded) |

**Changed since those results, in software, not yet run on the robot** (`resolution_plan.md`): P3 (the foot point, the full-width lane ROI, steering in cm), P1/P2 (the sign on the side thread, one gradient pass), D2 (a sign counts after 6 votes a sighting: replayed over the 28 Oct and 4 Nov runs it removes every false sign stop and keeps every real one). The steps to verify them are `phase1_robot_procedure.md`.
