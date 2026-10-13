# Question: What does the robot do when it loses the lane?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only what happens when the camera stops seeing the lane
lines. Nothing else about the robot is in scope.**

## The short answer

It doesn't guess, and it doesn't pretend. Every frame, estimation labels
the lane position it reports:

- **vision:** measured this frame.
- **hold:** not seen, for up to 7 frames (about 350 ms). The last good
  value is repeated, and labelled as held.
- **stale:** not seen for longer. The number is old, and navigation must
  not steer by it.

## Steering blind

While the lane is lost, the gyro takes over the job of "which way am I
pointing". The heading tracker adds up the gyro's turn rate, minus its
measured bias (about −1.1 °/s at rest on this robot). The result is "how far
I've turned since I last saw the lane", not an absolute heading. Lane
keeping steers to hold that at zero.

Adding up a rate drifts: any error in the bias grows into an angle over
time. That's fine for a second, not for a minute.

## The end of the road

If the lane stays stale, the robot creeps at slow speed (0.30 duty, just
above the 0.25 where the motors stall), still holding its heading. After 1 s
of that it brakes and ends the run:

- **Finished**, if the route is done and its finish is the mat's edge.
- **Ended early**, a safety stop, if the route isn't done.

A lost lane during a stop, or in the middle of an intersection with no lane
lines, isn't counted as the end. Those rules hold the frame first.

## A recent fix (2026-10-12)

With one lane line visible, the robot estimates the lane's center at half a
lane's width from that line. That half-width was hand-set at 228 px; the
runs measured the lane at 350–357 px across, so it should be about 178.

At 228, every one-line frame read about 0.25 off its neighbors. That's a
big enough jump to trip the jump gate, so one dash seen alone could set off
a swerve, then hold, then stale, then the end-of-course creep. At 178 the
mismatch is 0.04–0.08.

## The open question

How long can the gyro hold a heading before its drift matters, and is 1 s
of creeping the right time to decide the course has ended?
