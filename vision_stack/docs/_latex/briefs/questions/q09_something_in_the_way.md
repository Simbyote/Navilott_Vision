# Question: How does the robot see something in its way?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only obstacle detection and what the robot does about an
obstacle. Nothing else about the robot is in scope.**

## The short answer

By looking for anything that isn't floor. The course is only floor (dark
mats to light gray) and white tape, and neither has a color of its own. So
in the middle of the frame, a pixel counts as "not floor" when it is either:

- **much brighter than the floor:** 60 gray levels above it; or
- **clearly colored:** at least 15 away from the scene's neutral color, in
  a color space (Lab) that separates color from brightness.

## The tricks

- **The floor is measured every frame.** It's taken as the 25th-percentile
  gray of the lane view, which is wider than the robot's path, so an object
  in the path leaves floor beside it. A fixed level failed: a cap of 70
  called the light-gray floor an obstacle (2026-10-12).
- **Color is relative.** The neutral is the whole frame's median color, not
  a fixed white. Under warm light by the windows, the whole floor read as
  colored, and the empty lane was an obstacle for a second after a can was
  lifted away.
- **Tape is erased by shape.** Tape is bright but thin. A morphological
  opening with a 47 px square removes anything narrower: lane lines (up to
  about 42 px), dashes and stop lines. What survives is solid and large.
- **Distance comes from the base.** An upright object meets the floor at
  its lowest row. That row's distance comes from the same table that
  measures stop lines.
- **Half resolution.** The loop is already at its frame budget, so masks
  are built on every second pixel each way, a quarter of the work. An
  obstacle is tens of pixels across.

## What the robot does

The obstacle rule is the robot's first rule, above stop signs. When the
voted obstacle's base is within 12 cm (or already below the view), it
brakes. It waits until nothing at all has been seen for 1 s before going
on: releasing on "not near" would creep it into the object one stop at a
time.

## What it can't see

- **Black or gray objects** with no color, no brighter than the floor.
- **Objects outside the middle of the frame.**
- **Possible false alarms:** glare, a light floor past the mat's edge, or a
  stop sign low in the path.

Most gates are a first guess. The color threshold was checked on one day's
three runs (1,001 clear frames, 199 with a can within about 20 cm): 10 or 15
gave no false alarm and caught every can frame; 20 missed 33, 30 missed 99.

## The open question

Obstacle avoidance was a stretch goal. Stopping is not avoiding. What would
going around take?
