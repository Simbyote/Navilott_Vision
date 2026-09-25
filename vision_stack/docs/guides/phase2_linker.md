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
python3 -m src.phase2_linker --frames src/tests/data/frames --scale 2
```

| Option | Effect |
| --- | --- |
| `--camera` / `--video PATH` / `--frames DIR` | Frame source (pick one) |
| `--limit N` | Stop after N frames |
| `--fps N` | Capture rate for `--camera`; replay rate for `--frames`. Videos default to their own rate |
| `--width`, `--height` | Capture size; defaults to `params.py` (480×270) |
| `--views stop,traffic` | Extra views beside the lane view, which is always on |
| `--hsv PATH` | Calibrated HSV ranges; switches the color branch on. The traffic view is empty without it |
| `--stop-threshold C`, `--traffic-threshold C` | Confidence the next stage needs; candidates below it show amber in that view |
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

- Tuning is `MEASURED` from `phase2_linker.py`, the same the robot uses
- The color branch is off unless `--hsv` is given
- Undistortion is off; there's no option to turn it on from here yet
- The per-contour sign and traffic traces are on, since this is the debug entry point

## 4. Reading the results

Each run writes one folder, printed at the start as `output`:

```
runs/<YYYYMMDD_HHMMSS>/
    run.avi, run.csv            lane view: annotated video and per-frame decision log
    run_stop.avi, .csv          one pair per extra view
    run_traffic.avi, .csv
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
| `[TIMING]` | Median and p95 per stage, and the total as FPS. The total must stay under 50 ms for 20 FPS |

In the lane view: green candidates are usable, red ones carry the name of the gate that rejected them. Cyan and magenta mark the chosen left and right boundaries, gray the robot at ROI center, and yellow the implied lane center (red when the offset is pinned at ±1).

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
| `unknown view ...` | Only `stop` and `traffic` are valid for `--views` |
| Traffic view is empty | Add `--hsv calibration/hsv_ranges.json` |
| `capture failed: ... consecutive failed reads` | The camera stopped delivering frames; stop other camera processes and rerun |
| Timing well over 50 ms | Check `[TIMING]` for the stage responsible; `--stride` and `--no-display` reduce recording and display cost |
| `No module named src` or `cv2` | Not in `vision_stack/`, or environment not active |

Report any other error with the full terminal output.
