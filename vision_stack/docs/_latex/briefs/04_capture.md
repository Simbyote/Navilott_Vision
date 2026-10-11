# Brief: Capture

**This brief covers only the Capture section: how light becomes a numbered,
timestamped frame in Python. It does not cover what the robot does with the
frame; later sections do.**

## The section's main idea

Before vision can do anything, the camera has to deliver the newest frame
quickly, upright, and with an identity every later stage carries. Capture
makes no judgment about what's in the frame.

## The path

M12 lens → IMX290 sensor (1920x1080, 10-bit) → RAW Bayer over CSI-2 → the
Pi's ISP (debayer, scale) → libcamera → GStreamer (libcamerasrc →
videoconvert → videoflip → appsink) → OpenCV → a 270 x 480 x 3 BGR array in
Python.

## Decisions, and what goes wrong without each

- **The sensor mode is pinned** to the full 1920x1080 mode. Otherwise
  libcamera picks a mode from the output size, and different modes crop
  differently. That would change the field of view and invalidate the lens
  calibration.
- **The output is 480x270**, 16:9 like the sensor, so the ISP scales
  without cropping or stretching. It replaced 480x360 when the IMX219 was
  swapped for the IMX290.
- **Only the newest frame is kept** (drop=true, max-buffers=1). A slow
  frame means the next read is the current view, not a backlog: for
  steering, a skipped frame beats a late one.
- **The flip happens in GStreamer**, because the camera is mounted upside
  down. Every consumer sees the same upright frame.
- **Frame ids and timestamps are assigned in one place.** Ids are
  gap-free; timestamps come from a monotonic clock, since the Pi's wall
  clock is set by hand (NTP is blocked on the school Wi-Fi).
- **Exposure and white balance are the camera's own.** In a dim room the
  auto exposure brightened the scene until the traffic lamps blew out to
  white (2026-10-04).

## Failures

One failed read is skipped without using an id. About 1 s of consecutive
failures (20 reads) raises an error: a dead camera stops the run, a dropped
frame doesn't. That's requirement R1, verified in software.

## Still unmeasured

- The delay from exposure to the timestamp.
- Capture timing on the IMX290 (the old numbers were the IMX219's).
- Whether exposure should be locked at the venue.

## Questions the author has to answer here

1. Which of these decisions were made on purpose, and which were found by
   breaking something?
2. What did the IMX219 to IMX290 swap cost and gain?
