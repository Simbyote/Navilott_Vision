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
GStreamer                        libcamerasrc → videoconvert → videoflip → appsink
  │
  ▼
cv2.VideoCapture (CAP_GSTREAMER)
  │  (270, 480, 3) uint8 BGR
  ▼
CameraSource.read() → FrameData
```

The pipeline string, built by `build_gst_pipeline()`:

```
libcamerasrc sensor-config="sensor/config,width=1920,height=1080,depth=10" !
video/x-raw,width=480,height=270,framerate=20/1 !
videoconvert !
videoflip method=rotate-180 !
video/x-raw,format=BGR !
appsink drop=true max-buffers=1 sync=false
```

---

## Design decisions

**The sensor mode is pinned.** Without `sensor-config`, libcamera picks a mode from the requested output size, and different modes crop the sensor differently. Pinning the full 1920×1080 mode means the field of view doesn't change when the output size does. It also means a lens calibration stays valid, since it's tied to the mode, output size, flip and focus it was captured with.

**Output is 480×270.** That's 16:9, matching the sensor, so the ISP scales without cropping or stretching. It replaced 480×360 when the IMX219 was swapped for the IMX290.

**The image is flipped in GStreamer, not in Python.** The camera is mounted upside down (`CAMERA_ROTATE_180`). Flipping in the pipeline means every consumer, from the perception stages to recordings and the calibration script, sees the same upright frame.

**Only the newest frame is kept.** `drop=true max-buffers=1` makes the appsink throw away frames the pipeline didn't get to. If a frame takes too long to process, the next read gets the current view of the course instead of a backlog. For steering, a skipped frame is better than a late one.

**Frames arrive as BGR.** That's what OpenCV expects everywhere downstream. Older docs described YUV frames; the pipeline no longer uses YUV.

**Frame identity starts here and nowhere else.** `frame_id` and `timestamp_ms` are assigned in `CameraSource.read()`. No later stage creates its own. Every result dataclass copies them forward, and stages that combine two inputs check that the stamps match (`utils.check_same_frame`). A mismatch raises instead of labeling detections with the wrong frame.

**Timestamps are monotonic.** `time.monotonic_ns()` can't jump when the wall clock is set. That matters here: NTP is blocked on the school Wi-Fi, so the clock is set by hand with `timedatectl` before runs. A timestamp only means something as a difference between two frames in the same run.

**Ids are gap-free.** A failed read returns `None` and doesn't use up an id, so the caller can `continue` without creating a gap. The counter also survives `release()` and `open()`, so a camera reopened after a recovery never reuses an id that's already in the logs.

**Isolated failures are absorbed; a dead camera raises.** `read()` tolerates up to `max_consecutive_failures` failed reads in a row, which defaults to `fps` (about 1 s of dead camera), then raises `CaptureError`. One dropped frame doesn't stop the robot; a camera that has actually died does.

**The frame-rate band is advisory.** `CameraSource` warns outside `MIN_FPS`–`MAX_FPS` (5–30) but still runs. Below 5 FPS, control updates are too far apart for lane following. Above 30, the Pi Zero 2 W can't keep up with capture plus processing.

---

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
| `MIN_FPS`, `MAX_FPS` | 5, 30 | Warning band |
| `CAMERA_CALIB_PATH` | `calibration/camera_calib.json` | Lens calibration for this mode |

If you change the camera, the mode, the output size or the flip, the lens calibration is invalid. `test_calibration.py` catches the mismatch; see `guides/test_calibration.md`.

---

## Other frame sources

Everything above `CameraSource` also accepts replayed frames, so perception and estimation can be tuned off the robot and repeatably:

| Source (`src/debugger/live_view.py`) | Input | Timestamps |
| --- | --- | --- |
| `CameraFrameSource` | Live camera through `CameraSource` | Monotonic, from capture |
| `VideoFrameSource` | A recorded clip, e.g. a `live_view` `run.avi` | Nominal frame rate |
| `DirectoryFrameSource` | Image sequence, sorted by filename | Nominal frame rate |

In pytest, `--replay=DIR` does the same for the hardware tests (`guides/pytest.md`).

`VideoSink` writes frames to a video file. It must be opened before writing and rejects frames whose size doesn't match.

---

## Lens calibration

The M12 lens has noticeable barrel distortion toward the edges. Calibration belongs to the camera, but the correction runs in Phase 2 (`preprocess.undistort()`) so both branches see the same geometry.

- Capture and solve: `python3 -m src.scripts.calibrate_camera` (see `guides/calibrate_camera.md`)
- Verify an existing calibration: `guides/test_calibration.md`
- It's valid only for the sensor mode, output size, flip and focus it was captured with

---

## Open items

- **Camera choice.** Other cameras are being tested. Whichever is chosen means updating `SENSOR_CONFIG`, `FRAME_W`/`FRAME_H`, possibly `CAMERA_ROTATE_180`, and recalibrating.
- **Capture timing.** Re-measure with `pytest --hardware -k capture` on the current camera. Numbers from the IMX219 at 480×360 no longer apply.
- **Exposure.** The pipeline string sets no exposure or shutter controls, so libcamera runs auto-exposure. If motion blur shows up at driving speed, a fixed shorter shutter is the first thing to try.
