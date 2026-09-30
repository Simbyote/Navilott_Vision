# What the Tests Cover

A catalog of `src/tests/`: what each test file targets, what its software tests check, and what its hardware test records. How to run them is in `pytest.md`; how to interpret recorded runs is in `analysis.md`.

## How the suite is organized

Every test file answers one of three questions:

1. **Does each piece work correctly?** Groups 1–3: pipeline stages, peripherals, debug views.
2. **How does the whole system perform on real hardware?** Group 4: system runs.
3. **Do the analysis tools compute correctly?** Group 5: interpreter tests.

Most files hold both kinds of test:

- **Software** tests run anywhere, on synthetic or recorded input with known answers. They pass or fail, and they prove behavior: shapes, stamps, gate logic, math. They say nothing about how well the robot performs on the course.
- **Hardware** tests run on the Pi against the camera or a `--replay` dataset. They record measurements to `artifacts/` and fail only on broken contracts; performance problems are warnings. They measure; they don't judge.

A dash (—) means the file has no test of that kind.

## 1. Pipeline stages

In the order a frame passes through them.

| Test | Covers | Software checks | Hardware records |
| --- | --- | --- | --- |
| `test_capture` | `capture/camera.py`, frame delivery | Frame source contract against a scripted fake camera | Per-frame timing, effective FPS, dropped reads, sample frames; with `--record`, the replay dataset every other test can use |
| `test_calibration` | `scripts/calibrate_camera.py`, `preprocess.undistort` | Solver recovers a known synthetic lens; error paths; undistortion matches a reference; the team's `camera_calib.json` is valid and was solved from frames covering the whole image | Undistortion time per frame; straightness raw vs corrected on boards the solver never saw. Details: `test_calibration.md` |
| `test_preprocess` | `perception/preprocess.py` | Grayscale and blur on synthetic frames with known answers, and on every recorded frame if a dataset exists | Stage time per frame, sample images, latency histogram |
| `test_roi_crop` | `perception/roi_crop.py` | Lane, sign and traffic crops obey their bounds for any valid bounds and frame size | Crop time, the rects actually used, the crops, the debug overlay |
| `test_geometry` | `perception/geometry.py`, lane, stop-line and sign shapes | Tape, stop lines and red octagons drawn at known positions are found; red signs on gray, black and white floors, dim and small, with and without clutter; a sign-free ROI holds at `min_redness` and finds nothing; redness saturates rather than wraps; only the largest red blob is gated; closing rejoins a sign cut by a thin line; the production sign path matches the debug one; stop lines touching lane lines, broken by wide tape, tilted, or running off the ROI bottom; the lane detector's horizontal-line filter separates a touching stop line from the lane lines, removes its stubs, keeps tape ends and steeper lines, and leaves the stop-line detector's edges alone; each gate rejects what it should; reject counts add up; lane and sign results don't depend on the stop-line config | Stage time, reject counts per gate, edge maps, the sign mask, overlays, sign trace |
| `test_ground` | `perception/ground.py`, `scripts/calibrate_ground.py` | px → cm → px round trip and the centerline crossing on a known homography; fitting recovers it from its own corners, with and without noise; loading refuses a wrong size, alpha or lens calibration, undistortion off, a malformed or missing file, and never raises; the lens identifier follows only the values preprocess uses; corner ordering, the reference moved to the bottom of the view, and perspective-sized corner refinement; the script end to end on a board rendered on a known floor and distorted by the real lens | — |
| `test_stop_line_distance` | `perception/stop_line_distance.py` | Distance from the lane ROI bottom on hand-built candidates; `distance_cm` at the centerline crossing through a known homography, tilted, clipped, with the ROI origin added, and None without one or at another frame size; the nearest confident line is chosen; the confidence gate; nothing found reports no numbers; debug and production versions agree; a nearer line measures smaller through the real chain; `phase3_linker` writes the stop-line columns | — |
| `test_config` | `config.py`, `tests/scenes.py` | `MEASURED` undistorts, loads `hsv_ranges.json` and keeps the swept lane gates; `SCENE_CONFIG` is `MEASURED` without undistortion; both linkers run the one `MEASURED`, and `phase3_linker` the one `MEASURED_ESTIMATION` with its flags on top; importing the config loads no linker or debugger | — |
| `test_color_branch` | `perception/color_branch.py`, traffic lights | Colored blobs at known positions are found; gate wiring; HSV file loading | Stage time, mask areas, reject counts per color, overlays; uses `calibration/hsv_ranges.json` when present |
| `test_feature_fusion` | `perception/feature_fusion.py` | Ordering, conflict resolution, ROI-local coordinates | Fusion time, per-class counts, overlays |
| `test_lane_offset` | `perception/lane_offset.py` | Gates, boundary pairing and single-sided rules on hand-built candidates; the sign convention; candidates on or against a stop line are skipped, lane lines crossing it are kept | Offset, mode and confidence per frame; debug log; anchor overlay |
| `test_phase2_out` | `perception/phase2_out.py`, the output packet | Every packaging rule (PR-1 to PR-4) and failure case (F1–F5) in the Phase 2 contract, stop-line results included | Packaging time, one JSON snapshot of the packet per sample frame |
| `test_production_parity` | Debug-free twins in `geometry`, `color_branch`, `lane_offset`, `stop_line_distance`, `feature_fusion` | The fast production functions return the same results as their debug-instrumented versions, field by field, on the shared scenes (stop lines included) and the gate sweep, under the bare defaults, `SCENE_CONFIG`, `ALT_CONFIG` and `MEASURED` | — |
| `test_pipeline` | `pipeline.py`, the robot's flow | `Pipeline.perceive()` gives the same `Phase2Output` as `run_chain()`, field by field, on every shared scene and the gate sweep under five configs; `estimate()` and `step()` give the same packets and Phase 3 debug as `phase3_linker` over `drive_sequence()`, Phase 2 checked first on every frame; the drive moves every vote, lane status and the heading; each `ALT_CONFIG` group and `ALT_ESTIMATION` field changes the output, so a stage ignoring its config fails; the cm-scale lane width matches the linker's and holds frames to 480×270; no debug function runs; stage timing is off unless asked for, and adds `phase3` when on; the stamp is carried and the frame untouched. Stop lines clear of the lane lines leave the lane offset alone; ones touching the lane lines keep the lane (two boundaries at the right place) all the way up to the line, for 6–30 px tape, and are measured; the drive toggles the stop-line vote down to distance 0 | — |
| `test_estimation` | `estimation.py`, Phase 3 | Each estimation stage alone (the stop-line vote and held distance included), then `Phase3Processor` for ordering, stamps and pass-through, the wheel counts per second included; `SensorSample.from_frames` combines the IMU and encoder snapshots and keeps a stopped wheel as 0.0 | — |
| `test_drive` | `peripherals/drive.py`, motors and encoders | Quadrature decoding and direction per wheel, counts per second over each snapshot window (steady speed, after a reset, stopped reads 0.0), TB6612 direction and PWM commands, stop, short brake (both direction pins high, standby up, the wheel model stops), and the closed-loop / differential routines, against a fake pigpio | Spins each wheel with the wheels off the ground and checks the counts |
| `test_estimation_debug` | `estimation_debug.py`, Phase 3's debug twin | Packets and log identical to `Phase3Processor` over a randomized Phase 2 stream and the synthetic drive; a detection accepted vs gated (inclusive gate), vote raw / buffer / state; a lane jump rejected, held and gone stale, then re-seeded; the other lane reasons; a stop-line distance held; heading reset; every stage timed | — |

## 2. Peripherals

| Test | Covers | Software checks | Hardware records |
| --- | --- | --- | --- |
| `test_imu` | `peripherals/imu.py`, MPU-6050 | Accumulator, calibration and worker thread against a scripted fake sensor | Samples per window and stationary yaw noise. **Robot still** |
| `test_system` | `peripherals/system.py`, button and display | Debounce, the non-blocking `button_pressed()` (once per press, bounces ignored, the start press not counted twice), countdown, MM:SS formatting and cleanup against fake `pigpio` and `tm1637` | Drives the real display (`rdy`, countdown, clock) and reads the button at rest. **Watch the display** |
| `test_system_monitor` | `debugger/system_monitor.py`, Pi health | Temperature, clock, throttle-flag and memory parsing against fake `/sys` and `/proc` files; the sampling thread | One real sample: temperature, clock and memory must read |

## 3. Debug views

The views watched while tuning. These tests make sure what they show is true.

| Test | Covers | Software checks | Hardware records |
| --- | --- | --- | --- |
| `test_debug_lane` | `debugger/debug_lane.py` | Overlays land at known pixels; gate labels match every real `lane_offset` gate | Every debug image each stage produces for 3 sample frames, plus a video of the run. Start here when tuning by eye |
| `test_debug_stop` | `debugger/debug_stop.py` | Contour grading into pass / low / reject, labels, CSV row, summary | The stop view as video + CSV, 3 rendered stills |
| `test_debug_stopline` | `debugger/debug_stopline.py`, and the lane view's stop-line label | Top-edge grading into pass / low / reject, lane skips read from lane offset's log, the band geometry, every scene renders, CSV row, summary, `--views stopline` end to end; the lane view labels skipped candidates "on stop line" | — |
| `test_debug_lanegeo` | `debugger/debug_lanegeo.py`, and geometry's lane contour trace | One trace entry per contour agreeing with the reject counts, each gate named with what it measured, no trace unless asked; candidates graded with lane offset's gates; removed-edge count; red / amber / green in the ROI panel; every scene renders, with and without a trace; CSV row, summary, `--views lanegeo` end to end | — |
| `test_debug_phase3` | `debugger/debug_phase3.py` | One frame size for every state and scale; the first frame, no detections, no lane result, None and held distances; passed / gated boxes in their colors; lane bar colored by status; the 5 s timeline; CSV row and summary | — |
| `test_phase3_linker` | `phase3_linker.py`: video and timing | The video and its CSV on by default, one frame per processed frame; `--no-video` leaves `p3.csv` unchanged; render timed but kept out of the budget and the total; Ctrl-C still closes the writer; the linker runs the traced processor; `--encoders` starts the reader against a fake pigpio, writes each wheel's counts per second, and cancels the callbacks at the end | — |
| `test_maneuver` | `maneuver.py`, the drive trial's state machine | Against a simulated robot (`sim_robot.py`): every step in order and the body turned to the target; the gyro bias and accel baseline measured at rest and replacing a wrong configured bias; the yaw sign found under either IMU convention, and a dead drivetrain named; a dragging wheel corrected by each term alone and both; corrections clamped; the leg time cap; the turn's slow band, overshoot FAIL, timeout; every still step brakes, and braking holds the turn inside a tolerance coasting misses; `hold`: the four holds in order, braking, counts recorded, the same results as without holds, an unanswered hold trips no safety stop, hold time off the run limit, an early press can't skip a hold; every safety stop (late frame, stall, turn past 270°, run time, outside abort, no IMU) | — |
| `test_navigation` | `navigation.py`, the Navigation contract, and its checks (`navigation_checks.py`) | Each `Command` rule named, and every broken rule listed; `BRAKE` is a zero-duty brake and the default command coasts; the `Navigator` interface; a navigator that keeps the contract passes every check, and navigators that each break one rule (overdrive, stall, brake with duty, not a Command, driving on a stale lane or on stop, steering away, never driving) are caught by theirs; steering checked both ways; the checks reset first and feed every frame | — |
| `test_maneuver_linker` | `maneuver_linker.py` | The whole run folder on the simulated robot; `maneuver.csv` lines up with `p3.csv` and logs exactly the commands sent; vision records but never steers; lane re-acquisition after the turn; the motors stop first on finishing, Ctrl-C, an error (traceback saved, then raised) and the camera ending; the recorder drops instead of blocking and copies each frame; start button hooks; flag and `--set` overrides; render-only; `--hold` end to end, resume asked only while holding, the button or Enter resuming | — |
| `test_debug_maneuver` | `debugger/debug_maneuver.py` | The strip for every step at one size; leg and turn bars; red aborts; `render_run` on a real run folder, with missing frames and a records file cut short by a crash | — |
| `test_debug_traffic` | `debugger/debug_traffic.py` | Blob grading, per-color totals, mask panel, labels | The traffic view as video + CSV, 3 rendered stills |
| `test_debug_video` | `debugger/debug_video.py` | Recording, stride, sidecar CSV, codec failure, the shared grading | — |
| `test_live_view` | `debugger/live_view.py`, the bench runner | Frame sources and stamps, `stages.csv`, headless display, early exits, command line | A full bench run: every view's video and CSV, `stages.csv`, `summary.txt` |

## 4. System runs

Not tied to one module: these run the whole chain.

| Test | Covers | Software checks | Hardware records |
| --- | --- | --- | --- |
| `test_stage_timing` | Phases 1–2 against the frame budget; `analysis/stage_timing.py` | The timing breakdown math on tables with known answers; that it knows every stage `run_chain` times | Every stage's time, the capture wait and the loop time per frame; `timing_budget.png`, `timing_per_frame.png` |
| `test_soak` | Phases 2–3 over a long run; `analysis/soak.py` | Heat, throttling, memory growth and slowdown found in synthetic logs where they were planted | Temperature, CPU clock, throttle flags and memory each second beside every frame's timing. Runs only with `--soak-minutes=N` |

## 5. Interpreter tests

These test the tools in `src/analysis/`, not the robot. The tools are run on recorded data (`analysis.md`); these tests only check that their math is right on data with known answers. All software.

| Test | Checks the tool that... |
| --- | --- |
| `test_jitter` | finds frame-interval tails, over-budget streaks and periodic spikes |
| `test_stability` | measures offset noise and lane-mode flicker on a still scene |
| `test_offset_accuracy` | judges measured offsets against true positions and the ±2 cm spec |
| `test_detection_range` | finds how far away stop signs and traffic lights are reliably detected |
| `test_gate_rejections` | ranks which detector gate discards the most candidates |
| `test_state_timeline` | measures how long Phase 3 states last and what they change into |
| `test_compare_runs` | lists every value that changed between two runs |
| `test_common` | covers the shared CSV reading, run finding and statistics helpers |

The software halves of `test_stage_timing` and `test_soak` (group 4) belong here too.

## Supporting files

| File | Role |
| --- | --- |
| `test_utils.py` | Known answers for the shared helpers in `src/utils.py` |
| `conftest.py` | Not tests: the `--hardware`, `--replay`, `--frames`, `--soak-minutes` options, the frame source, and the artifact folders |
| `artifacts.py` | Not tests: how hardware tests write CSVs, JSON, images and videos |
| `scenes.py` | Not tests: the one source of synthetic frames (`synthetic_frame`, `scene`, `SCENES` with stop lines, the gate `SWEEP`), `drive_sequence()` (a stamped frame-and-sensor drive for Phase 3), `SCENE_CONFIG` (`MEASURED` without undistortion), `ALT_CONFIG` and `ALT_ESTIMATION` (every stage's and every Phase 3 field's tuning moved), and `same()`, the field-by-field comparison |

## Which test for which question

| Question | Evidence |
| --- | --- |
| Does the robot keep up with 20 FPS? | `test_stage_timing` (hardware), then `analysis/stage_timing`, `analysis/jitter` |
| Does it stay fast when hot? | `test_soak` (hardware), then `analysis/soak` |
| Is the lane offset within ±2 cm? | Recorded runs at measured positions, then `analysis/offset_accuracy` |
| How much warning before a stop sign? | Recorded runs at measured distances, then `analysis/detection_range` |
| Is the lens correction right? | `test_calibration` (hardware, held-out boards) |
| Why does a detector miss things? | `test_debug_lane` or `test_debug_stop` / `test_debug_traffic`, then `analysis/gate_rejections` |
| Is a stage's logic right? | That stage's software tests (group 1) |
| Did a change make things worse? | Two hardware runs, then `analysis/compare_runs` |
