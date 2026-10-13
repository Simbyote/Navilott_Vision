# Question: How does the robot know where it is in the lane?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how one camera frame becomes one steering number.
Nothing else about the robot is in scope.**

## The short answer

It finds the lane lines nearest the image center on each side, takes the
point midway between them as the lane's center, and measures how far the
image center (where the robot is) sits from it. That's the offset:

    offset = (image center − lane center) / half the view's width

**+ means the robot is right of center, so steer left.** The scale is half
the lane view's width (216 px at 480x270), a fixed scale for the steering.

## The details that matter

- **Where a line is measured:** at its foot, the mean position of its
  lowest few rows, nearest the robot. An angled line's middle sits halfway
  up the view and says less about where the robot is now.
- **Nearest, not outermost:** on a two-lane street, the outermost pair of
  lines would center the robot on the whole street. The nearest line on
  each side bounds its own lane.
- **Sanity checks:** a pair closer than 60 px is one marking seen twice; a
  pair wider than 400 px spans more than one lane. Either falls back to
  the strongest single line.
- **One line:** the center is projected half a lane from it. That
  half-width was hand-set at 228 px until 2026-10-12, when the runs measured
  the lane at 350–357 px across, so 178. At 228, one-line frames read about
  0.25 off their neighbors, enough to set off a swerve.
- **Stop lines aren't lane lines:** lines across the lane are removed from
  the lane detector's edges first. A stop line touching both lane lines
  would otherwise join them into one shape too wide to be a lane line, and
  the lane would vanish while the stop line was in view. Early runs drifted
  right and failed at intersections for reasons like this.

## How good is it?

Requirement P3 asks for the robot's position to within ±2 cm. A routine
exists to check it, parking the robot at ruler-measured offsets, but it
hasn't been run on the robot yet.

## The open question

The camera has to sit exactly on the robot's centerline for "image center
= robot" to hold. How would you know if it didn't?
