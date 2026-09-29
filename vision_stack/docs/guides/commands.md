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
python3 -m src.phase3_linker --camera --gyro-bias DPS         # apply bench gyro bias
python3 -m src.phase3_linker --camera --no-display --no-video # text and timing only
```

Output: `runs/p3_<YYYYMMDD_HHMMSS>/` (`p3.csv`, `summary.txt`)

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
