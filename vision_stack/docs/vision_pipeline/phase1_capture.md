# Phase 1: Camera Acquisition

> Photons to timestamped frames.

Phase 1 turns the camera into a stream of `FrameData`: one BGR frame plus the id and timestamp that every later stage carries. It makes no judgment about what's in the frame. Its only jobs are to deliver the newest frame quickly and to be the single source of frame identity.

**Code:** `src/capture/camera.py` · **Constants:** `src/params.py` · **Tests:** `src/tests/test_capture.py`

---

## Hardware path

```
Course
  │  light
  ▼
M12 lens → IMX290 sensor         1920×1080, 10-bit, pinned mode
  │  RAW Bayer over CSI-2
  ▼
Pi ISP (debayer, scale)
  │
  ▼
libcamera                        buffers, sensor timing, exposure
  │
  ▼
GStreamer                        libcamerasrc → appsink
  │
  ▼
cv2.VideoCapture (CAP_GSTREAMER)
  │  (270, 480, 3) uint8 BGR, upside down
  ▼
CameraSource.read() → orient() → FrameData      rotated 180° (CAMERA_ROTATE_180)
```

The pipeline string, built by `build_gst_pipeline()`:

```
libcamerasrc sensor-config="sensor/config,width=1920,height=1080,depth=10" !
video/x-raw,format=BGR,width=480,height=270,framerate=20/1 !
appsink drop=true max-buffers=1 sync=false
```

**The turn upright is in OpenCV, not GStreamer.** The ISP hands over BGR, so nothing in the pipeline copies the frame before the appsink. A `videoflip` there read the ISP's buffer itself, back to front a pixel at a time, from memory the ARM doesn't cache: GStreamer's thread (`task0`) took 100% of a core (pi-load, 2026-10-20) against 25–35% when `videoconvert` still made the first copy. `orient()` rotates the frame the appsink already copied into ordinary memory, one pass. `calibrate_camera` turns its frames the same way, so a lens calibration matches what the robot sees.

---

## Design decisions

**The sensor mode is pinned.** Without `sensor-config`, libcamera picks a mode from the requested output size, and different modes crop the sensor differently. Pinning the full 1920×1080 mode means the field of view doesn't change when the output size does. It also means a lens calibration stays valid, since it's tied to the mode, output size, flip and focus it was captured with.

**Output is 480×270.** That's 16:9, matching the sensor, so the ISP scales without cropping or stretching. It replaced 480×360 when the IMX219 was swapped for the IMX290.

**The image is turned upright once, at capture.** The camera is mounted upside down (`CAMERA_ROTATE_180`). `CameraSource.read()` turns every frame before handing it on, so every consumer, from the perception stages to recordings and the calibration script, sees the same upright frame. Every calibration (lens, ground, stop-line table) is tied to that orientation: changing `CAMERA_ROTATE_180` means recalibrating.

**Exposure and white balance are the camera's own unless set.** With `params.CAMERA_CONTROLS` empty, libcamera's auto exposure and auto white balance judge the whole frame and settle wherever the scene, and where they started, leads them; so the same sign can read red one run and not the next with nothing in the room moved. In a dim room the auto exposure blew the traffic lamps out to white (2026-10-04); backlit by windows, it darkened a stop sign and the white balance turned it purple, its redness (R − max(G, B)) at or under 0, so no `min_redness` could find it (2026-10-26). Any setting also changes how the lanes and the light look, so re-check them with it. "Camera controls" below has what can be set and how to try it.

**Only the newest frame is kept.** `drop=true max-buffers=1` makes the appsink throw away frames the pipeline didn't get to. If a frame takes too long to process, the next read gets the current view of the course instead of a backlog. For steering, a skipped frame is better than a late one.

**Frames arrive as BGR.** That's what OpenCV expects everywhere downstream. Older docs described YUV frames; the pipeline no longer uses YUV.

**Frame identity starts here and nowhere else.** `frame_id` and `timestamp_ms` are assigned in `CameraSource.read()`. No later stage creates its own. Every result dataclass copies them forward, and stages that combine two inputs check that the stamps match (`utils.check_same_frame`). A mismatch raises instead of labeling detections with the wrong frame.

**Timestamps are monotonic.** `time.monotonic_ns()` can't jump when the wall clock is set. That matters here: NTP is blocked on the school Wi-Fi, so the clock is set by hand with `timedatectl` before runs. A timestamp only means something as a difference between two frames in the same run.

**Ids are gap-free.** A failed read returns `None` and doesn't use up an id, so the caller can `continue` without creating a gap. The counter also survives `release()` and `open()`, so a camera reopened after a recovery never reuses an id that's already in the logs.

**Isolated failures are absorbed; a dead camera raises.** `read()` tolerates up to `max_consecutive_failures` failed reads in a row, which defaults to `fps` (about 1 s of dead camera), then raises `CaptureError`. One dropped frame doesn't stop the robot; a camera that has actually died does.

**The frame-rate band is advisory.** `CameraSource` warns outside `MIN_FPS`–`MAX_FPS` (5–30) but still runs. Below 5 FPS, control updates are too far apart for lane following. Above 30, the Pi Zero 2 W can't keep up with capture plus processing.

---

## Camera controls

`CAMERA_CONTROLS` passes libcamerasrc controls as properties on the element. These are the ones the robot's camera takes (`gst-inspect-1.0 libcamerasrc`, libcamera v0.7.2); `capture/camera.CAMERA_CONTROLS_SPEC` holds them with their values, and an unknown name or a bad value fails by name when the source is built, not as GStreamer's "unable to start pipeline". `make camera-controls` prints them.

| Control | Values | What it does |
| --- | --- | --- |
| `awb-mode` | auto, incandescent, tungsten, fluorescent, indoor, daylight, cloudy, custom | The color temperatures auto white balance picks from; a preset bounds it, so window daylight can't drag the colors as far |
| `awb-enable` | true / false | Auto white balance on; off is untried here, and there's no colour-gains control to set the gains by hand |
| `exposure-value` | −8 to 8 | Auto exposure's target in stops: +1 twice as bright. The first thing to try for a dark sign against bright windows |
| `ae-constraint-mode` | normal, highlight, shadows, custom | What auto exposure protects: `highlight` keeps bright parts (windows, LEDs) from clipping, `shadows` lifts dark ones |
| `ae-exposure-mode` | normal, short, long, custom | How exposure is split between shutter and gain: `short` favors a fast shutter, less motion blur |
| `exposure-time-mode`, `exposure-time` | auto / manual; µs | Manual holds the shutter; at 20 FPS at most ~50 000 µs |
| `analogue-gain-mode`, `analogue-gain` | auto / manual; 1.0 up | Manual holds the sensor gain: brighter and noisier as it rises |
| `digital-gain` | 1.0 up | Gain after the sensor |
| `ae-enable` | true / false | Auto exposure on; for fixed exposure use the two manual modes instead |
| `ae-flicker-period` | µs | Mains flicker to avoid: 8333 under 60 Hz lights, 10 000 under 50 Hz |
| `saturation` | 0 to 32, 1.0 normal | Color strength: above 1 a washed-out red reads redder |
| `sharpness` | 0 to 16, 1.0 normal | Edge sharpening |

There's no brightness, contrast, metering mode or colour-gains control on this camera's libcamerasrc.

**Trying them.** Every command that opens the camera takes `--camera-control KEY=VALUE` (repeatable) over `CAMERA_CONTROLS` for one run, and make passes `CAM="awb-mode=daylight saturation=1.5"` to them all (`phase2`, `phase3`, the driving linkers, `diag-nav`, `capture-anatomy`, `frame-meta`; `make run` uses `params` only). To compare settings, park the robot where the detection fails and run `make sweep-camera` (`scripts/sweep_camera.py`): each setting opens the camera, drops 3 s while the auto loops settle, and runs Phase 2 with `MEASURED` on 40 frames, then `runs/camera_<time>/summary.txt` ranks them by the stop sign (share at Phase 3's gate, and how red the sign ROI got), with the light and the lane beside it and `sheet.png` a picture of each. Without `CAM_TRY` it tries a set for backlight (white balance presets, `exposure-value=1`, `shadows`, `saturation=1.5`); `--rounds 2` repeats the list, since window light changes by the minute. `make frame-meta` records what the auto loops chose per frame (exposure time, gains, colour gains, colour temperature, lux), and `make camera-meta` graphs it on one time axis, with when auto exposure settled and how far it swung after; `pytest --hardware` records three seconds of it, which `make hw-report` graphs too. Two recordings from the same spot, one where a detection failed and one where it worked, show what the camera changed. Keep the winner in `CAMERA_CONTROLS`, then check the lanes and the light on a run.

## Contract: `FrameData`

| Field | Type | Meaning |
| --- | --- | --- |
| `frame` | `(H, W, 3) uint8` | BGR, upright, as the appsink delivered it |
| `frame_id` | `int` | 0-based count of delivered frames; unique and gap-free for the life of the `CameraSource` |
| `timestamp_ms` | `int` | Monotonic ms at the appsink pull; meaningful only as a difference within one run |

`FrameData` is frozen. Only `CameraSource.read()` creates one on the robot. The replay sources and `run_chain()` build them from recorded frames.

---

## Configuration

All in `src/params.py`:

| Constant | Value | Notes |
| --- | --- | --- |
| `SENSOR_CONFIG` | 1920×1080, 10-bit | Pinned sensor mode |
| `CAMERA_ROTATE_180` | `True` | Camera mounted upside down |
| `FRAME_W`, `FRAME_H` | 480, 270 | Output size |
| `FPS` | 20 | Requested frame rate |
| `CAMERA_CONTROLS` | `{}` | libcamerasrc controls ("Camera controls" above); empty is the camera's auto |
| `MIN_FPS`, `MAX_FPS` | 5, 30 | Warning band |
| `CAMERA_CALIB_PATH` | `calibration/camera_calibration.json` | Lens calibration for this mode |

If you change the camera, the mode, the output size or the flip, the lens calibration is invalid. `preprocess.undistort()` raises on a calibration whose image size doesn't match the frame; a new mode, flip or focus at the same size it can't see, so recalibrate.

---

## Other frame sources

Everything above `CameraSource` also accepts replayed frames, so perception and estimation can be tuned off the robot and repeatably:

| Source (`src/debugger/live_view.py`) | Input | Timestamps |
| --- | --- | --- |
| `CameraFrameSource` | Live camera through `CameraSource` | Monotonic, from capture |
| `VideoFrameSource` | A recorded clip, e.g. a `live_view` `run.avi` | Nominal frame rate |
| `DirectoryFrameSource` | Image sequence, sorted by filename | Nominal frame rate |

In pytest, `--replay=DIR` does the same for the hardware tests.

`VideoSink` writes frames to a video file. It must be opened before writing and rejects frames whose size doesn't match.

---

## Lens calibration

The M12 lens has noticeable barrel distortion toward the edges. Calibration belongs to the camera, but the correction runs in Phase 2 (`preprocess.undistort()`) so both branches see the same geometry.

- Capture, solve and verify: `make calib-camera` (`python3 -m src.scripts.calibrate_camera [capture|solve|verify|all]`; `--help` has the steps)
- Verify an existing calibration: `python3 -m src.scripts.calibrate_camera verify`, which undistorts saved frames and scores how much straighter the board got; `pytest --hardware src/tests/test_calibration.py` times the undistortion per frame
- It's valid only for the sensor mode, output size, flip and focus it was captured with

---

## Open items

- **Camera choice.** The IMX290 is the robot's camera. Changing it again means updating `SENSOR_CONFIG`, `FRAME_W`/`FRAME_H`, possibly `CAMERA_ROTATE_180`, and recalibrating.
- **Capture timing.** Re-measure with `pytest --hardware -k capture` on the current camera, or see the whole capture path with `make capture-anatomy` and graph it with `make capture-report` (each step's cost per frame, and how old a frame is when Python gets it). Numbers from the IMX219 at 480×360 no longer apply.
- **Exposure.** `params.CAMERA_CONTROLS` is empty, so libcamera runs its own auto exposure and white balance; `make sweep-camera` compares settings where a detection fails. In a dim room the auto exposure blew the lamps out to white (2026-10-04). Locking exposure and white balance at the venue is the main lighting fix (`concepts.md`, "Lighting"); if motion blur shows up at driving speed, a fixed shorter shutter is the first thing to try.
