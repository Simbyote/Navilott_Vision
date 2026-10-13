# Question: How does the robot stop at a line it can't see?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how the robot finds and stops at a stop line. Nothing
else about the robot is in scope.**

## The short answer

It can't see the line when it reaches it, so it remembers it. The camera's
view of the floor ends about 10 cm ahead of the robot, so a stop line leaves
the view before the robot gets there. A small tracker follows the line
through three phases:

1. **Approach:** the line is in view.
2. **Crossing:** the line has left the bottom of the view. This counts as
   one intersection.
3. **Reached:** some time later, the robot is "at" the line. A stop sign or
   red light brakes it here.

A line that vanishes while still far up the image isn't the robot driving
over it; it's flicker or a lost detection. Only a line last seen within 25
rows of the bottom counts as passed.

## The story: one number, four values

The time from "left the view" to "reached" is STOP_DELAY_MS:
- **0 at first:** braking the moment the line left the view stopped the
  robot short of it.
- **1500 ms** carried it a few cm past the line.
- **500 ms** stopped it at the line (2026-10-05).
- **0 again on 2026-10-07:** the course got a traffic light hanging over the
  center of the intersection, and 500 ms of driving carried the robot under
  it and out of view. Turns then needed a 1 s "advance" into the
  intersection first, since the stop lines were moving back into the street.
- **500 ms again on 2026-10-12,** with the advance removed. The commit gives
  no reason; the comment in the code still describes the 0.

## Subtleties

- **A delay is a time, not a distance.** The same 500 ms covers less floor
  on a tired battery.
- **The crossing phase lasts at least one frame,** so the robot is
  "reached" at least one frame (about 50 ms) after the line leaves view.
- **Double counting:** a line that flickers back into view during a
  crossing could count as a second intersection. One rule prevents it:
  while a crossing is active, the tracker takes up no new line.

## The open question

Why did 0 ms go back to 500 ms, and does the hanging light still stay in
view? The delay was tuned on one mat, at one speed, on one battery level.
Will it hold on the day?
