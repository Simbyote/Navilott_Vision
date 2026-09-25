# Running the Tests (pytest)

Every module in `src/` has a matching `src/tests/test_<module>.py`. Each test runs in one of two modes:

- **software**: contract and known-answer tests on synthetic or recorded input. No camera or robot needed; pass/fail. This is the default.
- **hardware**: characterization on the Pi against the real camera, IMU or display. Writes CSVs, images, videos and summaries to `artifacts/` for review.

Tests outside the selected mode are deselected, not skipped, so the output lists only what ran.

## Requirements

- The project's virtual environment, with `pytest` installed
- Hardware mode only:
  - camera not in use by `phase2_linker`, `live_view` or `rpicam-hello`
  - `sudo pigpiod` running (display and start-button tests)
  - I2C enabled (IMU test)
- `matplotlib` is optional. Without it, histogram images are skipped and the CSVs carry the same data.

## 1. Setup

```
cd ~/Navilott_Vision/vision_stack
source .venv/bin/activate
```

Run everything from `vision_stack/`, the folder holding `pytest.ini`. That file sets the import path and points pytest at `src/tests`, so no path arguments are needed.

On the Pi, set the clock first. Artifact folders are named by timestamp and are wrong until it's fixed.

```
timedatectl                                   # check Local time and Time zone
sudo timedatectl set-ntp false                # required before set-time
sudo timedatectl set-time "YYYY-MM-DD HH:MM:SS"
timedatectl                                   # confirm
```

## 2. Software tests (any machine)

```
pytest
```

Expect a summary like `605 passed, 10 skipped`. Skips are normal until their input exists, and `pytest.ini` prints the reason for each:

| Skip reason | Goes away when |
| --- | --- |
| `no recorded dataset in tests/data/frames` | A dataset is recorded (step 3) |
| `no camera_calib.json yet` | The camera is calibrated (`calibrate_camera.md`) |

Narrower runs:

```
pytest src/tests/test_geometry.py        # one module
pytest -k "sign and not overlay"         # tests whose names match
pytest --lf                              # only what failed last time
pytest -x                                # stop at the first failure
```

Run this after every code change. The full suite takes about 15 s on a laptop, most of it in `test_calibration.py`.

## 3. Hardware tests (on the Pi)

```
pytest --hardware
```

Runs every hardware test against the live camera for 100 frames each.

Give options that take a value with `=`, as in `--frames=300`. With a space, pytest reads the value as a folder of tests to run and then doesn't recognize the options at all.

| Option | Effect |
| --- | --- |
| `--frames=300` | Frames per test (default 100) |
| `--replay=src/tests/data/frames` | Feed recorded frames instead of the camera |
| `--record` | The capture test also saves its frames as the dataset software tests replay |
| `--artifact-dir=DIR` | Write results somewhere other than `./artifacts` |
| `--software --hardware` | Both modes in one run |

Record a dataset first, on the course with the robot where it will drive:

```
pytest --hardware --record --frames=300 -k capture
```

After that, every other hardware test can run from the recording, off the camera and repeatably:

```
pytest --hardware --replay=src/tests/data/frames
```

To run one hardware test, add its file:

```
pytest --hardware src/tests/test_imu.py
```

### What each hardware test needs and writes

| Test file | Needs | Look at |
| --- | --- | --- |
| `test_capture` | Camera | `frames.csv`, `summary.json` (effective FPS, dropped reads), sample frames |
| `test_preprocess`, `test_roi_crop`, `test_geometry`, `test_color_branch`, `test_feature_fusion`, `test_lane_offset`, `test_phase2_out` | Camera or `--replay` | Per-stage timing CSV and histogram, sample images, `summary.json` |
| `test_debug_lane` | Camera or `--replay` | Every image each stage produces for 3 sample frames, plus `lane_annotated.avi`. Start here when tuning a detector by eye |
| `test_debug_stop`, `test_debug_traffic` | Camera or `--replay` | That view as video + CSV, 3 rendered stills, summary |
| `test_live_view` | Camera or `--replay` | A full bench run: one video and CSV per view, `stages.csv`, `summary.txt` |
| `test_calibration` | `camera_calib.json`; checkerboard in view | Raw vs undistorted images, straightness per region. Use frames the solver never saw |
| `test_imu` | MPU-6050; **robot still** | Samples per window, stationary yaw noise |
| `test_system` | `pigpiod`; **watch the display** | Shows `rdy`, counts down, ticks the clock; checks the button reads low at rest |

`test_color_branch`, `test_debug_traffic` and `test_live_view` use `calibration/hsv_ranges.json` when it exists and the uncalibrated scaffold otherwise; their summaries say which.

## 4. Results

Each hardware run creates one folder:

```
artifacts/<YYYYMMDD_HHMMSS>/
    run_meta.json          platform, OpenCV version, git commit and dirty flag, camera target, options
    <test name>/           that test's CSVs, images, videos and summary
```

Hardware tests assert only contracts (stamps carried, shapes, counts that add up) and a few hard lines, such as "undistortion must not make a board less straight". Everything else is data to review, not a gate. A low frame rate, for example, prints a warning rather than failing.

To share a run:

```
tar -czf test_results.tgz artifacts/<YYYYMMDD_HHMMSS>
```

Send the archive along with the terminal output.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named src` or `cv2` | Not in `vision_stack/`, or environment not active |
| `camera unavailable` skips | Stop other camera processes; `rpicam-hello --list-cameras` must list imx290 |
| `IMU unavailable` skip | Enable I2C; check wiring and address 0x68 |
| `system peripherals unavailable` skip | `sudo pigpiod` |
| `unrecognized arguments: --hardware ...` | An option was given its value with a space (`--frames 300`, `--replay DIR`); use `--frames=300`, `--replay=DIR` |
| Hardware tests `deselected` | Add `--hardware`; software is the default |
| A software test fails after a code change | Run that file alone with `-x -vv`; the test name says which rule broke |

Report any other error with the full terminal output.
