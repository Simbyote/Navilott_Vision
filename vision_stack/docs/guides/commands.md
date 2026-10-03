# Command Reference

One line per command, grouped by task. Each section links to the guide that
explains the options, what gets written, and how to read the results.

Run everything from `~/Navilott_Vision/vision_stack` with the venv active:

```
cd ~/Navilott_Vision/vision_stack
source .venv/bin/activate
```

On the Pi, check the clock first so run folders get the right timestamp:

```
timedatectl                                     # check Local time and Time zone
sudo timedatectl set-ntp false                  # required before set-time
sudo timedatectl set-time "YYYY-MM-DD HH:MM:SS"
```

---

## Production run — [production_run.md](production_run.md)

The course run: route on the terminal, `St N` on the display, start button, time on the display. Nothing is recorded.

```
sudo pigpiod                                    # once per boot
python3 -m src.main                             # route.json, 300 s cap
python3 -m src.main --route my_route.json       # another route
```

Afterwards: the final time; `E  N` / time alternating = ended early at step N (Ctrl-C to exit); `E  t` = the run cap; `Err ` = an error (traceback on the terminal).

---

## Tests — [pytest.md](pytest.md), [tests.md](tests.md)

Software (any machine):

```
pytest                                          # full software suite
pytest src/tests/test_geometry.py               # one module
pytest -k "sign and not overlay"                # tests whose names match
pytest --lf                                     # only what failed last time
pytest -x                                       # stop at the first failure
```

Hardware (on the Pi):

```
pytest --hardware                               # everything, live camera
pytest --hardware --record --frames=300 -k capture    # record the replay dataset
pytest --hardware --replay=src/tests/data/frames      # hardware tests on recorded frames
pytest --hardware src/tests/test_imu.py         # IMU only
pytest --hardware src/tests/test_system_monitor.py src/tests/test_capture.py   # health
pytest --hardware src/tests/test_stage_timing.py      # per-stage timing
pytest --hardware --soak-minutes=15 src/tests/test_soak.py   # soak
```

Output: `artifacts/<YYYYMMDD_HHMMSS>/`

---

## Watch detections (Phase 2) — [phase2_linker.md](phase2_linker.md)

```
python3 -m src.phase2_linker --camera                 # live
python3 -m src.phase2_linker --video clip.mp4         # recorded video
python3 -m src.phase2_linker --frames DIR             # images, sorted by filename
python3 -m src.phase2_linker --camera --no-display --limit 600    # 30 s headless
python3 -m src.phase2_linker --frames src/tests/data/frames --scale 2
```

Extra debug views (`--views` takes any of `stop`, `traffic`, `stopline`):

```
python3 -m src.phase2_linker --camera --views stop,traffic
python3 -m src.phase2_linker --camera --views stopline --stopline-threshold 0.4
```

Also useful: `--stop-threshold C`, `--traffic-threshold C`, `--hsv PATH`,
`--stride N`, `--out DIR`.

Output: `runs/<YYYYMMDD_HHMMSS>/` (`run.avi`, `run_<view>.avi`, CSVs,
`stages.csv`, `summary.txt`)

---

## Check estimation (Phase 3) — [phase3_linker.md](phase3_linker.md)

```
python3 -m src.phase3_linker --camera                 # live
python3 -m src.phase3_linker --video clip.mp4
python3 -m src.phase3_linker --frames DIR
python3 -m src.phase3_linker --camera --imu --limit 600       # 30 s with the IMU
python3 -m src.phase3_linker --camera --imu --encoders       # IMU and wheel encoders (sudo pigpiod first)
python3 -m src.phase3_linker --frames src/tests/data/frames   # repeatable replay
python3 -m src.phase3_linker --camera --print-every 1         # every frame
python3 -m src.phase3_linker --camera --print-every 0         # events only
python3 -m src.phase3_linker --camera --cm-per-px S           # fill lane_offset_cm
python3 -m src.phase3_linker --camera --gyro-bias DPS         # override config.GYRO_BIAS_DPS
python3 -m src.phase3_linker --camera --no-display --no-video # text and timing only
```

Output: `runs/p3_<YYYYMMDD_HHMMSS>/` (`p3.csv`, `summary.txt`)

---

## Drive trial — [maneuver_linker.md](maneuver_linker.md)

The robot moves. Place it at the far end of the mat first; `sudo pigpiod` once per boot.

```
python3 -m src.maneuver_linker --no-motors --no-button    # bench check: nothing moves, stops at the yaw-sign check
python3 -m src.maneuver_linker --leg-counts 800           # first run on the mat: short leg, measure it
python3 -m src.maneuver_linker --hold                     # stop after each step to measure by hand
python3 -m src.maneuver_linker                            # full trial with the defaults (MANEUVER in config.py)
python3 -m src.maneuver_linker --set turn_slow_band_deg=30 --kp-heading 0.02
python3 -m src.maneuver_linker --render runs/maneuver_<YYYYMMDD_HHMMSS>   # rebuild a run's video
```

Output: `runs/maneuver_<YYYYMMDD_HHMMSS>/` (`summary.txt`, `report.json`, `maneuver.csv`, `p3.csv`, `maneuver.avi`)

---

## Intersections — [intersection_linker.md](intersection_linker.md)

One intersection per run with the whole chain driving; each judged PASS or CHECK.

```
python3 -m src.intersection_linker left --camera       # one left turn at a real stop line
python3 -m src.intersection_linker all --camera        # straight, left, right in turn
python3 -m src.intersection_linker straight --camera --no-motors   # bench: nothing moves
```

Output: `runs/intersection_<YYYYMMDD_HHMMSS>/` (`summary.txt`, `report.json`, one run folder per maneuver)

---

## Navigation — [navigation_linker.md](navigation_linker.md)

The whole chain drives the robot with `--camera`; replays never move it. `sudo pigpiod` once per boot.

```
python3 -m src.navigation_linker --camera --no-motors --no-button --max-run-s 10   # bench: nothing moves
python3 -m src.navigation_linker --camera --max-run-s 10    # short first run on the mat
python3 -m src.navigation_linker --camera                   # start button, 30 s cap
python3 -m src.navigation_linker --video runs/<run>.avi     # what it would have commanded
python3 -m src.navigation_linker --render runs/nav_<YYYYMMDD_HHMMSS>   # rebuild a run's video
python3 -m src.navigation_linker --camera --route my_route.json   # another route (default: route.json)
```

Output: `runs/nav_<YYYYMMDD_HHMMSS>/` (`summary.txt`, `report.json`, `nav.csv`, `p3.csv`, `nav.avi`)

The course plan is `route.json` in `vision_stack/`, read at startup; edit it between runs:

```json
{"maneuvers": ["left", "straight", "right"], "finish": "edge"}
```

`finish` is `edge` (the lane running out after the last maneuver) or `stop_line` (stop at the first stop line after it). Left and right turns run inside the intersection rule; prove each with `intersection_linker` below.

---

## Camera calibration — [calibrate_camera.md](calibrate_camera.md), [test_calibration.md](test_calibration.md)

```
rpicam-hello --list-cameras                           # must list imx290
python3 -m src.scripts.calibrate_camera               # capture, solve, verify in one go
python3 -m src.scripts.calibrate_camera capture --append
python3 -m src.scripts.calibrate_camera solve
python3 -m src.scripts.calibrate_camera verify
```

Check an existing calibration:

```
pytest src/tests/test_calibration.py -k calibration_file      # file checks
pytest src/tests/test_calibration.py
python3 -m src.scripts.calibrate_camera capture --frames holdout_frames --count 12
pytest --hardware --replay=holdout_frames --frames=12 src/tests/test_calibration.py
pytest --hardware --frames=300 src/tests/test_calibration.py  # move the board around
```

---

## Stop-line distance in cm — [calibrate_stop_line.md](calibrate_stop_line.md), [calibrate_ground.md](calibrate_ground.md)

Tape strips at measured distances, no checkerboard (lens calibration first):

```
python3 -m src.scripts.calibrate_stop_line                      # one mark at a time; Enter to fit
python3 -m src.scripts.calibrate_stop_line --marks 3:6.2,6:24.9,10:41.5,15:55   # refit, no camera
```

Or the full floor homography, from a checkerboard: `python3 -m src.scripts.calibrate_ground --square-cm=2.46 --origin-x-cm=-9.8`

---

## Pi diagnostics — [diagnostics.md](diagnostics.md)

Record a run's threads, cores, throttling and memory from a separate process (the run is unchanged):

```
python3 -m src.diagnostics.monitor -- python3 -m src.main
python3 -m src.diagnostics.monitor -- python3 -m src.navigation_linker --camera --no-motors --no-button --max-run-s 60
python3 -m src.diagnostics.monitor --match src.main          # attach to a run started elsewhere
top -H -p $(pgrep -f src.main)                               # live, per thread (names show)
```

Output: `runs/diag_<YYYYMMDD_HHMMSS>/` (`summary.txt`, `threads.csv`, `cores.csv`, `system.csv`, `meta.json`)

---

## Analyze runs — [analysis.md](analysis.md)

```
python3 -m src.analysis.<tool>                        # newest run with the right CSV
python3 -m src.analysis.<tool> <folder or csv>        # a specific run
python3 -m src.analysis.<tool> --help
```

| Tool | Input | Answers |
| --- | --- | --- |
| `stage_timing` | `artifacts/<run>` | where the frame time goes |
| `jitter` | `artifacts/<run>` | frame-to-frame timing spread |
| `gate_rejections` | `artifacts/<run>` | which gates reject candidates |
| `compare_runs` | `artifacts/<a> artifacts/<b>` | what changed between two runs |
| `stability` | `runs/<run>` | lane offset noise while still |
| `state_timeline` | `runs/<run>` | lane and drive state over time |
| `offset_accuracy` | `positions.csv` (`true_cm,run`) | P3 ±2 cm check |
| `detection_range` | `distances.csv` (`distance_cm,run`) | how far out detection holds |
| `soak` | `artifacts/<run>` | drift over a long run |
| `nav_run` | `runs/nav_<run>` or `runs/intersection_<run>/left` | what decided each frame, lane keeping, each intersection and the 2 s after, wheel balance, latency |

---

## Test procedure by tier — [testing_procedure.md](testing_procedure.md)

| Tier | Run |
| --- | --- |
| 0 Software | `pytest` |
| 1 Health | `pytest --hardware src/tests/test_system_monitor.py src/tests/test_capture.py` |
| 2 Characterization | replay + `test_stage_timing.py`, then `stage_timing`, `jitter`, `gate_rejections`, `compare_runs` |
| 3 Scenarios | `phase3_linker --camera --limit=… --out=runs/…`, then `stability`, `state_timeline`, `offset_accuracy`, `detection_range` |
| 4 Soak | `pytest --hardware --soak-minutes=15 src/tests/test_soak.py`, then `soak` |

---

## Getting results off the Pi

```
tar -czf run_results.tgz runs/<YYYYMMDD_HHMMSS>
tar -czf test_results.tgz artifacts/<YYYYMMDD_HHMMSS>
scp -r <user>@<pi-host>:~/Navilott_Vision/vision_stack/runs/<YYYYMMDD_HHMMSS> .
```
