# Brief: Perception

**This brief covers only the Perception section: what the robot finds in a
single frame and how. It does not cover smoothing across frames
(Estimation) or driving decisions (Navigation).**

## The section's main idea

Perception answers four questions about one frame, with no memory of the
frame before. Where are the lane lines? Is there a stop line, and how far
ahead? Is there a lit traffic light, and what color? Is there a stop sign?
It uses classical computer vision (thresholds, edges, shapes, color bands),
not a learned model.

## Why classical vision

The course is structured: white tape on dark mats, one octagon, three lamp
colors. Thresholds and shapes handle that on a Pi Zero 2 W with predictable
timing and memory, and every decision traces to a number that can be tuned
on the course. The cost: sensitivity to lighting, and every new situation
needs a new rule.

## The detectors

- **Lane lines:** Canny edges on the bottom of the frame. Lines across the
  lane are removed first, so a stop line can't join the lane lines into one
  shape. Then contours are traced and gated by area, shape and brightness.
  The lane is the nearest line on each side of the image center.
  - **Offset:** + means the robot is right of center, so steer left.
  - **One line visible:** the lane center is projected from it, using a
    hand-set half-lane width.
- **Stop lines:** edges whose brightness gradient points up or down (a
  lane line's sides point left or right). The top and bottom edges of the
  tape are paired into one line. Gates: length at least 60 px; tilt, at
  most 2° on the robot since 2026-10-07 (5° let parts of the intersection
  pass as stop lines, each counted as another intersection); thickness;
  brightness.
  - **Distance:** in rows from the bottom of the view, and in cm through a
    ground homography or a tape-measured table.
- **Traffic lights:** the course's lights are bare LEDs, and an unlit
  colored lens is as bright as a lit one's colored ring. So "glow mode"
  looks for what only a lit LED does: clip the camera to white. It finds
  the white spots and names each by the color of the ring around it. On
  196 labelled frames at the stop (2026-10-06): all right.
- **Stop signs:** a "redness" image (red minus the larger of green and
  blue), thresholded, then the largest red blob is checked for 8 or 9
  corners and solidity. Not yet tuned on real course frames.

## Seeing why a frame failed

Every gate counts its rejections, and debug views draw each candidate with
the gate that refused it, so "why did this frame go blind?" is answered
without re-running.

## Questions the author has to answer here

1. Which detector was hardest to get right, and what was the turning
   point?
2. What does the 2° stop-line gate give up? A crooked approach scores low:
   a 1.5° line scores about 0.51 against 0.70 level.
3. How well tuned is each detector on real frames, and how do you know?
