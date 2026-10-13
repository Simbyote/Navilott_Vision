# Brief: Estimation

**This brief covers only the Estimation section: how per-frame detections
and sensor readings become one trusted record per frame. It does not cover
how detections are made (Perception) or what the robot does with the
record (Navigation).**

## The section's main idea

Perception says what one frame shows; estimation decides what to believe.
The line between them is memory. Anything that uses history (smoothing,
voting, holding) or sensors belongs here; judgments about one frame's pixels
don't. Estimation never invents a measurement. When vision drops out, it
says so and how long it has been.

## The parts

- **Lane filter:**
  - A frame counts as a measurement only if a lane was found and its offset
    is within 0.5 of the current estimate. A bigger jump between frames
    isn't physical at driving speed.
  - Measurements are smoothed with an exponential moving average
    (α = 0.35).
  - **Status:** *vision* (measured this frame), *hold* (dropout, up to 7
    frames, about 350 ms; the last value is repeated) or *stale* (longer;
    navigation must not steer by it). That's requirement R2.
- **Heading tracker:**
  - While vision is lost, it integrates the gyro's yaw rate, minus its bias
    (about −1.1 °/s at rest on this robot).
  - The result is how far the robot has turned since the last good frame,
    not an absolute heading. Integration drifts with any bias error.
- **Voting:**
  - Each traffic-light state, stop sign, stop line and obstacle must win 2 of
    the last 3 frames to change state. One bad frame can't flip a decision; that's
    requirement R3.
  - Confidence gates come first: 0.40 for lights, 0.45 for signs.
  - The cost is about one frame (50 ms) of delay.
- **Sensors:** a background thread reads the IMU and wheel encoders
  together at 100 Hz and hands each frame its batch. Estimation never
  imports a hardware driver, so it runs on a laptop from recordings.

## The packet

One EstimationPacket per frame goes to navigation. Its fields:

- **Lane:** `lane_offset` (+ = right of center) and `lane_status`.
- **Heading:** `heading_error`.
- **Decisions:** `drive_state` (go / caution / stop), `stop_sign_detected`,
  `stop_line_detected` with its distance.
- **Pass-throughs:** yaw rate, wheel counts per second, and the frame's
  lane mode.
- The frame's id and timestamp.

## Questions the author has to answer here

1. Why three frames for the vote, and not two or five?
2. Why 7 frames of hold? What happens to the robot in those 350 ms?
3. What does a 0.0 pass-through mean: zero, or unavailable?
