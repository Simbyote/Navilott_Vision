# Question: How does the robot turn at the right corner without a map?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how the robot knows which way to go at each
intersection and how it makes the turn. Nothing else about the robot is in
scope.**

## The short answer

It counts. A route file (route.json) lists one maneuver per intersection,
for example left, straight, right, and how the run finishes. Every stop line
that passes under the camera is the next intersection, so the robot never
needs to know where it is, only how many intersections it has passed.

## Why lane keeping hands over

Inside an intersection, the crossing street's lane lines pull the lane
estimate sideways (2026-10-01 runs). So from the moment the stop line leaves
the view, an intersection rule drives instead, in stages:

1. **To the line:** straight, holding the heading on the gyro.
2. **Turn** (left or right only): fixed wheel commands until the gyro reads
   85° turned that way. That's 90° less the coast after the wheels stop: a
   180° test turn needed a 171° target. A time limit (4.1 s left, 2.4 s
   right) ends the turn if the gyro never gets there.
3. **Exit:** straight on the new heading until the lane is back:
   - both lane lines for 10 frames;
   - or one line for 20;
   - or 4 s of driving.

## The arcs

- **Left:** a wide arc into the far lane. Each wheel's command is a duty, a
  share of the battery voltage. It's (0.50, 0.73) now; it was (0.36, 0.63)
  when first measured on 2026-10-01.
- **Right:** a tight arc around the right wheel, (0.50, 0.25).

## What can go wrong

- **One miscount shifts every later turn.** A missed or false stop line
  does it, which is why the stop-line detector's tilt limit was tightened
  to 2°.
- **Each crossing blocks new stop lines while it lasts.** Otherwise a mark
  on the far side would count as another intersection.
- **The exit can end too early, or too late.** At 3 frames the next lane's
  stop line was taken up right after a crossing, so it became 10.

## The open question

The robot trusts its count completely. What would it take to notice it had
miscounted?
