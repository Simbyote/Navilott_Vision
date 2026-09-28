# Ground-Plane Calibration

Fits the homography that turns a point in the undistorted frame into centimeters on the floor, and writes `calibration/ground_homography.json`. With it loaded, the stop-line distance is reported in cm (`distance_cm`) as well as in lane-ROI rows (`distance_px`). Without it, everything runs and the cm fields are empty.

The camera is rigidly mounted and the floor is flat, so one homography covers every floor point the camera sees. It is fit on the frames `preprocess_frame` produces with the robot's own settings (`MEASURED`), and is only valid for exactly those: the lens calibration, `undistort_alpha` and the 480×270 output. The file records all three, and the pipeline refuses it, with a warning, if any of them has changed.

## Requirements

- The lens calibration done first (`calibrate_camera.md`). The script refuses to run without it, since the fit has to be on undistorted frames.
- OpenCV's 10×7-square checkerboard (`pattern.png`, **9×6 inner corners**), printed at 100% on stiff paper or foam board.
- The robot on the floor in its driving pose, camera mount final.
- A ruler or calipers, and tape.

## Floor coordinates

| Axis | Direction | Origin |
| --- | --- | --- |
| X | right of the robot, + | the robot's centerline |
| Y | forward, + | the **robot reference point**: the floor at the bottom-center of the camera's view |

The reference point is where the camera's view of the floor begins, about 3 cm ahead of the robot, so a stop line the robot reaches reads 0 cm and there's room to stop. The script finds it from the fit itself; you don't have to measure it.

`distance_cm` is Y where the stop line's near edge crosses the robot's centerline (X = 0), which is the distance that matters for stopping, even when the line is seen at an angle.

## 1. Place the board

1. **Across the lane ROI.** The lane ROI is the bottom 30% of the view. Cover as much of it as the board allows, near the bottom and centered. Distances outside the board are extrapolated, and the script warns when it covers less than 60% of the ROI's width or height.
2. **Square to the robot.** The board's 9-corner side runs left to right across the view, its rows parallel to the robot's axle. The robot's X axis is defined by the board, so a board turned 5° turns every lateral measurement 5°. Line its edge up with a straight edge laid against both front wheels.
3. **Flat.** Tape all four corners down. A bowed board is the most common source of a high max error, and it bows the result without anything else looking wrong.
4. **Lit evenly**, with no glare on the squares.

## 2. Measure

- **Square size (`--square-cm`).** Measure across several squares and divide; don't trust the nominal print size, since printers scale. For example, 8 squares measuring 19.68 cm is 2.46 cm.
- **Corner 0's X (`--origin-x-cm`).** Corner 0 is the board's **far-left inner corner**, as the camera sees it: the top-left one in the image. Measure its distance from the robot's centerline, + to the right, - to the left. A tape along the centerline marked on the floor makes this easy.
- **Corner 0's Y (`--origin-y-cm`), optional.** Leave it out to put the reference at the bottom of the camera's view (recommended). Give it only to use a different, measured reference point: corner 0's distance ahead of that point.

## 3. Run

From `vision_stack/`:

```
python3 -m src.scripts.calibrate_ground --square-cm=2.46 --origin-x-cm=-9.8
```

It averages the corners over 10 camera frames (`--frames=N`), fits, and writes:

| File | Contents |
| --- | --- |
| `calibration/ground_homography.json` | `H`, the conditions it's valid for, the board and origin, the per-corner error, the corners themselves |
| `calibration/ground_debug.png` | the board at 3× with corner 0 ringed, the +X / +Y axes, the lane ROI, a projected 5 cm grid, the centerline (yellow) and the reference point |
| `calibration/ground_raw.png` | one raw frame, to refit later without the camera |

To refit from saved frames: `--image=calibration/ground_raw.png` (repeatable).

The camera finds corners with a refinement window sized to the squares: perspective squeezes a 2.5 cm square to about 7 px at the far edge of the lane ROI, and the lens calibration's fixed 11×11 window would pull those corners onto the wrong edges.

## 4. Check before trusting it

Open `ground_debug.png`:

- **Corner 0** is ringed at the board's far-left inner corner. If it's anywhere else, the board is turned; turn it and rerun.
- **+X** points right along the board, **+Y** away from the robot.
- **The 5 cm grid** sits square on the board, every second line on a square edge (with 2.5 cm squares), and the yellow centerline runs up the middle of the robot's path.

And the printed summary:

| Output | Pass |
| --- | --- |
| `reprojection` | GOOD (max < 0.5 cm) or OK (max < 1.0 cm) |
| `corner 0` | Its X matches what you measured; its Y is plausible for where the board is |
| `view bottom` | About the width of floor you see at the bottom of the view |
| `board covers` | No warning, ideally |

Accuracy falls with distance: near the top of the lane ROI one pixel covers several millimeters of floor, so sub-pixel noise costs more there. Then place a strip of tape at a measured distance and check `line=<px>/<cm>` in `phase3_linker`'s status line.

## 5. When to redo it

- The camera mount, height or angle changed, or the camera was bumped
- `undistort_alpha` or the output size changed (the pipeline refuses the old file)
- The lens was recalibrated (the pipeline refuses the old file: it's tied to the lens calibration's values, not its date)

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `no lens calibration` | Run `calibrate_camera.md` first |
| `no 9x6 board found` | The whole board incl. its white margin in view; less glare; check it's the 10×7-square pattern |
| `board is turned` | Its 9-corner side has to run across the view; turn it a quarter turn |
| Reprojection POOR | Flatten and tape the board; measure `--square-cm` again; average more frames |
| Grid not square on the board | Board not square to the robot, or `--square-cm` wrong |
| `ground_homography.json not used (...)` at startup | The pipeline's settings changed since the fit; the reason is in the warning. Redo the fit |
| `distance_cm` always empty | The warning above, or no file yet |
