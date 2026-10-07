# Watching the Pipeline (phase2_linker)

Runs frames from the camera, a video or an image folder through Phases 1–2 and shows what the pipeline decided: every lane candidate, the gate that rejected it, the chosen boundaries and the offset. Each view is recorded as a video plus a CSV, with or without a display, so a run on the robot can be reviewed afterwards.

Use it to see why detection works or fails on a stretch of course. For what estimation does with the result, use `phase3_linker.md`. Neither one drives the motors.

## Requirements

- Camera not in use by `phase3_linker`, `rpicam-hello` or a pytest run (`--camera` only)
- A display is optional. Over ssh it runs headless and still records everything

## 1. Setup

```
cd ~/Navilott_Vision/vision_stack
source .venv/bin/activate
```

On the Pi, set the clock first. Run folders are named by timestamp and are wrong until it's fixed.

```
timedatectl                                   # check Local time and Time zone
sudo timedatectl set-ntp false                # required before set-time
sudo timedatectl set-time "YYYY-MM-DD HH:MM:SS"
timedatectl                                   # confirm
```

## 2. Run

One source is required:

```
python3 -m src.phase2_linker --camera                 # live
python3 -m src.phase2_linker --video clip.mp4         # a recorded video
python3 -m src.phase2_linker --frames DIR             # images, sorted by filename
```

Stop with `q` in the window, or Ctrl-C. Either way the recordings and summary are written.

Common variations:

```
python3 -m src.phase2_linker --camera --no-display --limit 600     # 30 s headless on the robot
python3 -m src.phase2_linker --camera --views stop,traffic         # add the stop and traffic views
python3 -m src.phase2_linker --camera --views stopline --stopline-threshold 0.4   # stop-line detection
python3 -m src.phase2_linker --camera --views lanegeo   # why a lane line was lost
python3 -m src.phase2_linker --frames src/tests/data/frames --scale 2
```

| Option | Effect |
| --- | --- |
| `--camera` / `--video PATH` / `--frames DIR` | Frame source (pick one) |
| `--limit N` | Stop after N frames |
| `--fps N` | Capture rate for `--camera`; replay rate for `--frames`. Videos default to their own rate |
| `--width`, `--height` | Capture size; defaults to `params.py` (480×270) |
| `--views stop,traffic,stopline,lanegeo` | Extra views beside the lane view, which is always on |
| `--hsv PATH` | HSV ranges to use instead of `calibration/hsv_ranges.json`, which `MEASURED` already loads |
| `--stop-threshold C`, `--traffic-threshold C`, `--stopline-threshold C` | Confidence the next stage needs; candidates below it show amber in that view. For stop lines that is `stop_line.min_confidence` (0.4) |
| `--stride N` | Record every Nth frame. Every frame is still processed and counted |
| `--scale N` | Magnify the overlay |
| `--no-display` | Force headless |
| `--out DIR` | Output folder instead of `runs/<timestamp>` |

With no arguments it prints the full help.

### Window keys

| Key | Action |
| --- | --- |
| `q` | Quit |
| space | Pause / resume |
| `s` | Save a still of the current view into the run folder |
| `v`, `1`–`9` | Next view / view by number |

## 3. What runs

- Tuning is `MEASURED` from `src/config.py`, the same the robot uses
- The color branch is on with `calibration/hsv_ranges.json`; `--hsv` swaps in other ranges
- Undistortion is on with `calibration/camera_calibration.json`
- The per-contour sign and traffic traces are on, since this is the debug entry point

## 4. Reading the results

Each run writes one folder, printed at the start as `output`:

```
runs/<YYYYMMDD_HHMMSS>/
    run.avi, run.csv            lane view: annotated video and per-frame decision log
    run_stop.avi, .csv          one pair per extra view
    run_traffic.avi, .csv
    run_stopline.avi, .csv
    run_lanegeo.avi, .csv
    stages.csv                  per-frame stage timings, lane mode and offset
    summary.txt                 the run's report
    still_<view>_NNN.png        stills saved with s
```

`summary.txt`, top to bottom:

| Section | Look for |
| --- | --- |
| Counts | `frames processed`, `dropped reads` (should be near 0), fused detections per frame |
| Lane | Mode histogram (how often `two_boundary` vs one-sided vs `none`), availability, and the longest blind runs |
| Stop / traffic | Per-gate rejection totals, and how many frames passed the threshold |
| Lane geometry | Mode histogram, frames where the horizontal-line filter removed edges, contours refused by geometry and candidates refused by lane offset, per gate |
| Stop line | Frames with a line through the gates and with a measured line, the distance range, lane candidates skipped as part of a stop line, top edges rejected per gate |
| `[TIMING]` | Median and p95 per stage, and the total as FPS. The total must stay under 50 ms for 20 FPS |

In the lane view: green candidates are usable, red ones carry the name of the gate that rejected them ("on stop line" when lane offset skipped them as part of a detected stop line). Cyan and magenta mark the chosen left and right boundaries, gray the robot at ROI center, and yellow the implied lane center (red when the offset is pinned at ±1).

In the stop view (`--views stop`): the left panel is the sign ROI in color with each red blob outlined by the gate that decided it: green passed, amber below `--stop-threshold`, red refused (`vert 6`, `sol 0.71`, `area 80`), and `smaller` for blobs set aside because only the largest is gated. The right panel is the redness image (R − max(G, B)): the sign should be the brightest thing in it, whatever the floor's brightness. The red mask is outlined in cyan and the threshold used is in its label (`mask > 62`). A missed sign with no bright shape on the right is a color problem (lighting, exposure, a faded sign); a bright shape that the cyan outline doesn't follow is a threshold problem; a cyan outline with a red label was refused by that gate. A threshold of exactly 20 means Otsu found nothing red and `min_redness` held it. `run_stop.csv` adds `rej_not_largest` and `red_threshold`.

In the traffic view (`--views traffic`) with MEASURED's glow mode: the left panel is the traffic ROI with every clipped-white spot boxed by the glow gate that decided it, and the right column is the white mask on top, then the red, yellow and green masks. Green is the light: its ring color, confidence, white pixels and ring share (`red c1.00 11px ring 91%`). Amber is below `--traffic-threshold`. Red boxes were refused:

- `white 1px`: too few white pixels.
- `asp 10.50`: the wrong shape, such as the board-edge glare strip.
- `no ring 6%`: white with no band holding at least 10% of its ring, such as glare. The figure is the best band's share.
- `yellow 2px smaller`: another spot had more white, such as a reflection.

How to read it:

- A lit LED with no box: nothing clipped white, so it's an exposure problem.
- `no ring` on a real LED: its ring's hue or saturation is outside the bands. The color masks show which band it misses.
- A red LED labelled yellow: the ring reads orange. Look at the red and yellow masks around it.

The header shows the best spot, and the footer shows the spots seen and the count per glow gate. `run_traffic.csv` adds `glow`, `mask_white_px`, `best_ring_share` and `rej_white` / `rej_shape` / `rej_ring` / `rej_smaller`. Blob mode (`glow` 0) keeps the area and aspect columns and leaves the glow ones blank.

In the lane-geometry view (`--views lanegeo`): the top panel is the lane ROI with every contour the lane detector traced, colored by the stage that decided it. Red contours were refused by geometry's own gates, labeled with the gate and what it measured (`area 1203`, `aspect 1.1`, `span 0.9`, `int 95`, `pts 4`); the lane view never shows these. Amber ones passed geometry and were refused by lane offset, labeled with its gate (`conf`, `prox`, `len`, `wid`, `int`, `on stop line`). Green ones are usable, labeled with confidence and width. Cyan and magenta ticks mark the chosen left and right boundaries at their foot, yellow the lane center, gray the robot; detected stop lines are boxed gray. The bottom panel is the edge map the contours came from: Canny edges in gray, those the horizontal-line filter removed (a stop line across the lane) in red, the ones kept in white, and what the closing step filled in between in blue. A missing lane line with no contour at all was lost in the edges (look for red where the tape should be); a red contour was refused by geometry, an amber one by lane offset. The header shows the lane result and how many edge pixels the filter removed; the footer, this frame's geometry gate counts. `run_lanegeo.csv` has the per-frame counts, gates and result.

In the stop-line view: the top panel is the lane ROI. Accepted stop lines are outlined as the band between their paired edges (green, or amber below `--stopline-threshold`) with confidence and thickness; rejected top edges are red lines labeled with their gate (`short` with the length, `tilt`, `unpaired`, `dim`); the measured line has a white outline and an arrow to the ROI bottom with its distance, in px and, with a ground homography (`calibrate_ground.md`), in cm (`ON LINE` when the robot is on it); lane candidates lane offset skipped are boxed amber. The bottom panel is the gradient split: every Canny edge in gray, the kept top edges (dark to bright going down) in cyan, bottom edges in magenta, with the fitted lines drawn over them. A missed stop line with no cyan and magenta in the bottom panel was lost before the gates (Canny or the tilt split); one with a red label was refused by that gate. `run_stopline.csv` has the per-frame counts, gates and measurement.

## 5. Recording footage for replay

`run.avi` has the overlay drawn on it, so it can't be fed back into the pipeline. To record clean frames for replay, use pytest:

```
pytest --hardware --record --frames=300 -k capture
```

That saves the frames to `src/tests/data/frames`, which both linkers can replay with `--frames`.

## 6. Results

```
tar -czf run_results.tgz runs/<YYYYMMDD_HHMMSS>
```

From another computer, the folder can also be copied directly:

```
scp -r <user>@<pi-host>:~/Navilott_Vision/vision_stack/runs/<YYYYMMDD_HHMMSS> .
```

Send the archive with the terminal output and a note on where on the course it was recorded.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Help text instead of a run | No source given; add `--camera`, `--video` or `--frames` |
| `source error: ...` | Camera busy or missing, or the file or folder doesn't exist. `rpicam-hello --list-cameras` must list imx290 |
| `no display server ...; continuing headless` | Normal over ssh; everything is still recorded |
| `unknown view ...` | Only `stop`, `traffic`, `stopline` and `lanegeo` are valid for `--views` |
| Traffic view is empty | Check `calibration/hsv_ranges.json` against course lighting; `--hsv` tries other ranges |
| `capture failed: ... consecutive failed reads` | The camera stopped delivering frames; stop other camera processes and rerun |
| Timing well over 50 ms | Check `[TIMING]` for the stage responsible; `--stride` and `--no-display` reduce recording and display cost |
| `No module named src` or `cv2` | Not in `vision_stack/`, or environment not active |

Report any other error with the full terminal output.
