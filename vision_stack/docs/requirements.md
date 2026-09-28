# Requirements and Verification

> What the vision system must do, how each requirement is checked, and where it stands.

Each requirement has one way to check it. A requirement is **Verified** only when that check has passed on the current hardware: the IMX290 camera at 480×270 on the Pi Zero 2 W. Results from the IMX219 at 480×360 are kept as history but no longer count.

Values marked **TBD** need to be taken from the Product Specification Report or agreed with the team.

---

## Performance

| ID | Requirement | How it's checked | Status |
| --- | --- | --- | --- |
| P1 | Process at least **20 frames per second**, sustained | `phase3_linker --camera --limit 1200` (60 s): "FPS effective" in `summary.txt` ≥ 20, and "over budget" near 0 | Not met: 15 FPS measured on the IMX290 at 480×270, with frames still over budget. |
| P2 | Deliver each navigation update within **50.0 ms** of the frame being captured | `phase3_linker`: p95 of P2 + P3 in `summary.txt`. Capture-to-appsink delay isn't measured yet; see note | Not verified. The IMX219 averaged about 49 ms |
| P3 | Report lateral position in the lane to within **±2 cm** | Place the robot at measured offsets (0, ±2, ±4 cm from lane center) and compare the reported offset in cm; see procedure below | Not verifiable yet: needs the ground scale (cm per px) measured first |
| P4 | Run entirely on board, with no network connection | By design: nothing in the pipeline uses the network | Met by design |

On P2: the pipeline stamps a frame when it reaches the application, after exposure and ISP processing. Time from exposure to that stamp is a few frame periods at most but hasn't been measured. If P2 means "light to decision", it needs a separate measurement, such as filming an LED next to the robot's display.

---

## Detection

| ID | Requirement | How it's checked | Status |
| --- | --- | --- | --- |
| D1 | Detect the lane boundaries of the robot's own lane | Course recording through `phase3_linker`: share of frames in `vision` status, and longest `hold` / `stale` runs. Target **TBD** | Runs; not tuned on the IMX290 |
| D2 | Detect a stop sign at the expected range | Recordings approaching each sign: first frame with the stop-sign flag set, and no flag on stretches with no sign. Range **TBD** | Code runs; not tuned on course frames |
| D3 | Classify a traffic light as red, yellow or green | Recordings at each light state: voted state matches the light. Range **TBD** | Code runs; off until HSV ranges are calibrated |
| D4 | Detect the stop line at an intersection, and how far ahead it is | Proposed: with `calibration/ground_homography.json` fitted (`guides/calibrate_ground.md`), park square to a stop line at tape-measured distances from the reference point (the floor at the bottom of the camera's view) to the line's near edge: 0 (on it), 5, 10, 15, 20 cm and the far edge of the lane ROI. 100 frames each through `phase3_linker`. Pass: at every distance the stop line is detected in ≥ 95% of frames and the median `p2_stop_line_cm` is within ±1.0 cm of the tape (±0.5 cm at 0–10 cm), with a frame-to-frame standard deviation under 0.5 cm; repeat at ±10° to the line; and no stop line on 300 frames of plain lane with dashed center line. Tolerances **TBD** with navigation's stopping needs | Implemented (geometry detection, `stop_line_distance`, Phase 3 vote, cm through the ground homography); not verified on the course |
| D5 | Detect intersections | **TBD** | Not implemented; planned as the scene state machine |

D5 was in earlier plans but has no code yet. Either schedule it or mark it out of scope for this semester in the PSR. D4's stop line is detected and measured; intersections as a scene state (D5) are not.

---

## Robustness

These are internal targets from the design, not PSR requirements. They're here so each has a check.

| ID | Target | How it's checked | Status |
| --- | --- | --- | --- |
| R1 | A single dropped frame never stops the robot; a dead camera is reported within about 1 s | `test_capture` (software) | Verified in software |
| R2 | A lost lane is held for at most about 350 ms, then reported as unusable | `test_estimation` (software) | Verified in software |
| R3 | No single frame can flip the traffic or stop-sign state | `test_estimation` (software) | Verified in software |
| R4 | CPU under 70% and memory under 400 MB with navigation running | Full run with navigation, observed with `top` | Not verified |
| R5 | The motors stop if control stops arriving | **TBD**, depends on navigation's watchdog | Not implemented |

---

## Procedure for P3 (lane offset accuracy)

1. **Measure the ground scale.** Park the robot centered in a straight lane and run `phase3_linker --camera --limit 100`. Note `L` and `R` in the status line. The ground scale is 14 cm ÷ (R − L) px, using the lane width measured the same way (center of one line to center of the other; see `course.md`).
2. **Record the noise floor.** Still parked centered, run `--limit 300` and note the offset `std` in `summary.txt`.
3. **Measure at known offsets.** Shift the robot 2 cm and 4 cm right of center, then left, measuring with a ruler from the lane center to the robot's centerline. At each position run `phase3_linker --camera --cm-per-px <scale> --limit 100` and note the mean `lane_offset_cm` in `p3.csv`.
4. **Pass:** at every position, the mean reported offset is within 2 cm of the measured one.

Record the results, with the date and commit, in the status column above.

---

## Status summary

| | Verified | Not yet | Not implemented |
| --- | --- | --- | --- |
| Performance | P4 | P1, P2, P3 | |
| Detection | | D1, D2, D3, D4 | D5 |
| Robustness | R1, R2, R3 | R4 | R5 |
