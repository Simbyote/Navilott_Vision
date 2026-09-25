# Verifying the Camera Calibration

Checks that `calibration/camera_calib.json` is valid for this camera and actually straightens the image, using `src/tests/test_calibration.py`. Run it after every calibration (`calibrate_camera.md`) and before trusting lane offsets on the course.

The check has two parts:

- **File checks** (software, a few seconds): the calibration matches the camera mode, its numbers are plausible, and it was solved from frames covering the whole image.
- **Held-out check** (hardware, on the Pi): boards the solver never saw are undistorted through the pipeline's own `preprocess.undistort()` and scored for straightness, raw vs corrected.

## Requirements

- `calibration/camera_calib.json` from a completed calibration
- `calib_frames/` still holding the frames that calibration was solved from (for the coverage check)
- The same printed 9 × 6 checkerboard used to calibrate
- Camera not in use by `phase2_linker`, `live_view` or `rpicam-hello`
- Lens focus and camera mount untouched since calibrating

## 1. Setup

```
cd ~/Navilott_Vision/vision_stack
source .venv/bin/activate
```

On the Pi, set the clock first so the results folder is named correctly.

```
timedatectl                                   # check Local time and Time zone
sudo timedatectl set-ntp false                # required before set-time
sudo timedatectl set-time "YYYY-MM-DD HH:MM:SS"
timedatectl                                   # confirm
```

## 2. File checks

```
pytest src/tests/test_calibration.py -k calibration_file
```

Three tests run. All three must pass:

| Test | Checks | If it fails |
| --- | --- | --- |
| `test_calibration_file_matches_this_camera_mode` | Image size, sensor mode and 180° flip match `params.py` | The camera settings changed after calibrating; recalibrate |
| `test_calibration_file_intrinsics_are_plausible` | Reprojection < 1.0 px; ≥ 10 frames; fx and fy within 3%; image center within 15% of the frame's middle | See Troubleshooting |
| `test_calibration_file_was_solved_from_frames_covering_the_whole_image` | Every row and column of the 3 × 3 grid has frames, and ≥ 7 of 9 cells do | Recapture with `capture --append`, holding the board in the empty cells the failure prints |

A skip is not a pass. `pytest.ini` prints why a test skipped:

- `no camera_calib.json yet`: calibrate first.
- `calib_frames no longer holds the frames this calibration used`: the coverage check can't run. Recalibrate, or restore the frames from `calib_results.tgz`.

To also confirm the calibration code itself still solves a known lens correctly (about 15 s, no hardware):

```
pytest src/tests/test_calibration.py
```

## 3. Held-out check

The calibration must be tested on boards it wasn't solved from; a lens model always fits its own frames. Capture a fresh set into a separate folder:

```
python3 -m src.scripts.calibrate_camera capture --frames holdout_frames --count 12
```

Same board handling as calibration: robot still, board upright 15–40 cm away, fully in view. **Prioritize edges and corners**; that's where distortion is and where the correction has to show. This only captures; it doesn't solve, so the calibration file is untouched.

Then score them:

```
pytest --hardware --replay=holdout_frames --frames=12 src/tests/test_calibration.py
```

Use `=` for `--replay` and `--frames`; with a space, pytest reads the value as a test folder.

Alternatively, score live, moving the board through the frame while the test runs (300 frames at 20 FPS is about 15 s):

```
pytest --hardware --frames=300 src/tests/test_calibration.py
```

## 4. Reading the results

The hardware test writes to `artifacts/<YYYYMMDD_HHMMSS>/test_undistortion_characterization/`:

| File | Contents |
| --- | --- |
| `summary.json` | `straightness_by_region`: raw vs undistorted bow per 3 × 3 region; `coverage`; `frames_with_board`; `stage_ms`, the undistort cost per frame |
| `undistort.csv` | Per frame: timing, region, raw and undistorted straightness (px) |
| `*_raw_vs_undistorted.png` | Side-by-side images |
| `config.json` | The calibration that was tested |

Pass criteria:

| Check | Pass |
| --- | --- |
| The test itself | Passes. It fails if undistortion leaves any board more than 0.5 px *less* straight than raw |
| Off-center regions in `straightness_by_region` | Undistorted mean below 0.6 × raw, or below 0.5 px (the same rule `calibrate_camera verify` uses) |
| `coverage` | Boards in the corner and edge cells, not just `middle-center`. A center-only run can't show the correction |
| Images | Board lines and course tape straight on the UNDISTORTED side, including at the edges |
| `stage_ms` | A few ms on the Pi; it's paid on every frame |

A board close to the frame edge can come back with no undistorted score: undistortion crops the edges, so the board may no longer be fully in view. Those frames are left out of the undistorted average. If most edge frames are missing, hold the board slightly further in from the edge and rerun step 3.

## 5. Results

```
tar -czf calib_check.tgz calibration/camera_calib.json holdout_frames artifacts/<YYYYMMDD_HHMMSS>
```

Send `calib_check.tgz` with the terminal output of steps 2 and 3.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Camera mode test fails on `image_size` | `FRAME_W` / `FRAME_H` changed since calibrating; recalibrate at the current size |
| Camera mode test fails on `sensor_config` or `rotate_180` | Sensor mode or flip in `params.py` changed; recalibrate |
| `rms ... recapture` | Board not flat or not held still; flatten it, hold each pose ~1 s, recalibrate |
| `fx and fy should agree` | Board printed at the wrong scale or warped; reprint at 100% on rigid backing |
| Image center check fails | Frames bunched in one area; recapture with full coverage |
| Coverage check fails | `capture --append` with the board in the empty cells, then `solve` and `verify` |
| Hardware test fails: undistorted worse than raw | Wrong calibration file, or the lens or mount moved since calibrating; recalibrate |
| Off-center regions barely improve | Too few edge frames in the calibration; add them with `capture --append` and re-solve |
| `camera unavailable` skip | Stop other camera processes; `rpicam-hello --list-cameras` must list imx290 |
| `unrecognized arguments: --hardware ...` | Use `--frames=N` and `--replay=DIR`, not a space |

Report any other error with the full terminal output.
