# AutoBot Unified Pipeline — `main_pipeline.py`

A sequential, single-process pipeline that links Phase 1 (camera acquisition),
Phase 2 (vision perception), and Phase 3 (navigation signal processing) into
one loop. One frame in, one `EstimationPacket` out, every cycle.

---

## What it does per frame

```
libcamera / GStreamer
       │  BGR frame
       ▼
  preprocess_frame()      histogram equalization (YCrCb) + Gaussian blur
       │
       ▼
    crop_rois()           splits frame into lane_roi / traffic_roi / sign_roi
       │                  (NumPy views — no copy)
       ├──────────────────────────────────────────┐
       ▼                                          ▼
extract_traffic_light_candidates()        run_geometry_branch()
  HSV threshold → blob filter               Canny → contours → lane + sign filter
  → TrafficLightCandidate list              → LaneCandidate + SignCandidate lists
       │                                          │
       └──────────────┬───────────────────────────┘
                      ▼
              fuse_detections()           normalizes candidates, resolves conflicts
                      │                  → DetectionObject list
                      ▼
           compute_lane_offset()          pixel-space lateral offset [-1, 1]
                      │                  → LaneOffsetResult
                      ▼
        package_phase2_output()           assembles Phase2Output container
                      │
                      ▼
          Phase3Processor.process()       EMA filter, motion consistency check,
                      │                  majority vote, IMU dead-reckoning fallback
                      ▼
            EstimationPacket              → stdout log line
```

---

## Output

The pipeline produces **two outputs**:

### 1. Terminal log (always on)

One line per frame written to stdout:

```
12:34:01 [INFO] f=0042  t=71.3ms  offset=-0.0821  head=-2.5°  drive=go     stop_sign=F  lane_mode=two_boundary
```

| Field        | Meaning                                                                                                |
|--------------|--------------------------------------------------------------------------------------------------------|
| `f=`         | Frame counter since startup                                                                            |
| `t=`         | Total frame processing time in ms                                                                      |
| `offset=`    | Lateral offset from lane center, normalized `[-1.0, 1.0]`. Negative = right of center, positive = left |
| `head=`      | Heading error in degrees. Vision-primary; IMU yaw integration fallback                                 |
| `drive=`     | `go` / `caution` / `stop` — majority-voted over last 3 frames                                          |
| `stop_sign=` | `T` / `F` — majority-voted stop sign presence                                                          |
| `lane_mode=` | `two_boundary` / `left_only` / `right_only` / `width_rejected` / `none`                                |

A `[WARNING]` line appears if the frame took longer than the configured budget (e.g. 66.6 ms at 15 FPS).

### 2. Debug video (opt-in)

Set `SAVE_VIDEO = True` at the top of the script. Each frame is written to
`output.avi` in the working directory with a HUD overlay showing offset,
heading, drive state, frame time, and a lateral offset bar at the bottom of
the frame. The file is finalized on clean shutdown (Ctrl-C is safe; `kill -9`
is not).

---

## How to run

```bash
python main_pipeline.py
```

Stop with **Ctrl-C**. The camera and video writer release in the
`finally` block.

---

## How to save the log

```bash
python main_pipeline.py 2>&1 | tee pipeline.log
```

Live output continues to appear in the terminal. Everything is also written
to `pipeline.log` in the current directory.

---

## How to retrieve data from the Pi

### Pull the log file

```bash
scp pi@<pi-ip>:~/autobot/pipeline.log ./
```

### Pull the debug video using windows `scp`

```bash
scp pi@<pi-ip>:~/autobot/output.avi ./
```

Replace `<pi-ip>` with the Pi's IP address and adjust the remote path if your
working directory differs. Both files sit in whatever directory you ran the
script from on the Pi.

>> It may be easier to just install the WindowsSCP plugin on the target computer and utilize the GUI instead

---

## Key configuration (top of `main_pipeline.py`)

| Parameter                        | Default       | Effect                                                                                |
|----------------------------------|---------------|---------------------------------------------------------------------------------------|
| `FRAME_WIDTH` / `FRAME_HEIGHT`   | `480` / `360` | Capture resolution. Reduce to `320×240` to cut frame time if budget is being exceeded |
| `FPS`                            | `15`          | Target frame rate. Loop budget is derived from this: `1000 / FPS` ms                  |
| `SAVE_VIDEO`                     | `False`       | Set `True` to write `output.avi`                                                      |
| `LANE_CONF_THRESHOLD`            | `0.30`        | Minimum confidence to use a lane boundary detection                                   |
| `TRAFFIC_CONF_THRESHOLD`         | `1.1`         | Effectively disabled (max is 1.0). Re-enable after HSV calibration                    |
| `SIGN_CONF_THRESHOLD`            | `1.1`         | Same — disabled until sign geometry is calibrated                                     |
| `MIN_LANE_WIDTH_PX`              | `150.0`       | Two-boundary detections narrower than this are rejected                               |

---

## Sensor stubs

IMU and encoder reads in `_read_sensors()` return `None` for all fields until
the hardware interface is calibrated. Current phase 3 implementation handles `None`
accordingly:

- No IMU → heading error falls back to vision-derived estimate only
- No encoder → motion consistency jump threshold is not speed-scaled
- No encoder → dead-reckoning holds the last known lane offset for up to
  `deadreck_max_frames` (default 10) frames before going stale

Replace the `None` returns in `_read_sensors()` with the pigpio / MPU-6050
reads when the hardware is available.

---

## Demo scope

Traffic light and stop sign detections execute every frame for timing
validity but their outputs are gated out of the navigation signal by setting
both confidence thresholds above 1.0. Once traffic light detection and stop sign
geometry are calibrated, re-enable by dropping them to operational values after
calibration:

```python
TRAFFIC_CONF_THRESHOLD = 0.40
SIGN_CONF_THRESHOLD    = 0.45
```

---

## Main Directory Structure

These files are the only ones required to run the demo; other files are
artifacts of the development process.

```
vision_stack/
    capture/
        capture.py
        decompose.py
    dummy/
        dummy_hsv_ranges.json
    src/
        color_branch.py
        estimation.py
        feature_fusion.py
        geometry.py
        lane_offset.py
        main_pipeline.py
        phase2_out.py
        preprocess.py
        roi_crop.py
```
