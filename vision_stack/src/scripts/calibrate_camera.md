# Camera Calibration (IMX290)

Measures lens distortion and writes `calibration/camera_calib.json`. The pipeline loads it and undistorts every frame in preprocessing.

## Requirements

- Printed checkerboard: [OpenCV `pattern.png`](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png), 9 × 6 inner corners, printed at 100% scale and taped flat to a rigid backing
- Camera not in use by `phase2_linker` or `rpicam-hello`

Do not adjust the lens focus or the camera mount after calibrating. Either change invalidates the calibration.

## 1. Setup

```
cd ~/Navilott_Vision
source .venv/bin/activate
```

The prompt shows `(.venv)` when the environment is active. All commands below run from this directory with the environment active.

Set the clock. The Pi has no battery-backed clock and school Wi-Fi blocks network time sync, so it boots with the date it last synced. File and run-folder timestamps are wrong until this is fixed:

```
timedatectl                                   # check Local time and Time zone
sudo timedatectl set-ntp false                # required before set-time
sudo timedatectl set-time "YYYY-MM-DD HH:MM:SS"
timedatectl                                   # confirm
```

Use the current local time. `Time zone` should read `America/Chicago`; if not, `sudo timedatectl set-timezone America/Chicago` first. Re-enable sync (`sudo timedatectl set-ntp true`) when on a network that allows it.

```
git pull
rpicam-hello --list-cameras    # must list imx290
```

## 2. Capture, solve, verify

```
python3 -m src.scripts.calibrate_camera
```

Runs capture, solve, and verify in sequence. Previous frames, if any, move to `calib_frames_prev/`.

Frames save automatically when the full board is visible and has moved since the last save. Capture stops at 20 frames or 5 minutes.

Board placement: the robot stays stationary. Hold the board upright in front of the lens, printed side facing the camera, 15–40 cm away. Do not lay the board flat on the floor or place the robot on it; floor-only views cover too few angles and regions for a valid calibration.

- Cover all 9 regions of the frame. Progress prints as `coverage n/9`.
- Prioritize edges and corners, keeping the entire board in view.
- Vary distance and tilt (up to ~30°).
- Hold each position ~1 s. Avoid glare.

If regions are missing after capture:

```
python3 -m src.scripts.calibrate_camera capture --append
python3 -m src.scripts.calibrate_camera solve
python3 -m src.scripts.calibrate_camera verify
```

## 3. Pass criteria

| Output | Pass |
| --- | --- |
| `reprojection` | GOOD (< 0.5 px) or OK (< 1.0 px) |
| `RESULT` | PASS |

`calib_verify/` contains RAW / UNDISTORTED comparisons. Board lines on the undistorted side should be straight at the edges. Point the robot at course tape before verify runs so `verify_live.png` shows lane lines.

## 4. Pipeline check

On the course, ~30 s:

```
python3 -m src.phase2_linker --camera
```

- No `calibration ... not found` warning.
- `preprocess` time in `summary.txt` within a few ms of the previous 3.4 ms.
- Straight tape stays straight across the recorded frame, including the edges.

## 5. Results

```
tar -czf calib_results.tgz calibration/camera_calib.json calib_frames calib_verify
```

Send `calib_results.tgz` and the run folder from step 4 (path printed as `output`), or commit `calibration/camera_calib.json` and push.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named cv2` or similar | Environment not active; run `source .venv/bin/activate` |
| `could not open the camera` | Stop other camera processes; reboot if it persists |
| Board never detected | Full board incl. white border in view; move closer; remove glare; confirm 9 × 6 pattern |
| Reprojection POOR | Flatten board, hold still longer, rerun step 2 |
| RESULT: CHECK | Add edge/corner frames with `capture --append`, then `solve` and `verify` |
| `calibrated at WxH but frames are WxH` | Pipeline resolution changed; recalibrate at the current resolution |

Report any other error with the full terminal output.
