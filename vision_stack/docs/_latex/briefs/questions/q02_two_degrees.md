# Question: Why can two degrees decide whether the robot gets lost?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how stop lines are told apart from other tape, and
why the angle limit matters. Nothing else about the robot is in scope.**

## The short answer

The robot finds its way by counting stop lines. The route file says "left
at the first intersection, straight at the second, right at the third", and
each stop line passing under the camera is the next intersection. One false
stop line shifts every later turn by one. A missed one does the same. So the
filter that decides "is this tape a stop line?" decides whether the route
still makes sense.

## How a stop line is recognized

A stop line lies across the lane; lane lines run along it. The detector
keeps only edges whose brightness changes going up or down the image, not
left to right. It pairs the top edge of a strip of tape with its bottom
edge and checks the pair:
- at least 60 px long, longer than any lane tape is wide;
- thick like tape;
- bright;
- close to horizontal: the tilt gate.

## The story of the tilt gate

- **20°** was the default. The near end of a thick diagonal lane line past
  an intersection passed as a stop line (2026-10-01 run).
- **15°, then 5°** (2026-10-05) cut the false lines further along the mat.
- **2°** (2026-10-07): at 5°, pieces of the intersection itself were still
  taken for stop lines, each counted as another intersection. At 2° that
  stopped.

## What 2° costs

- **The limit filters pixels too.** It's applied to individual edge pixels
  before any line is fitted, so a line much steeper than 2° doesn't get
  rejected; it never forms at all.
- **Tilt lowers the score.** Confidence drops as tilt approaches the limit:
  a line at 1.5° scores about 0.51 against 0.70 when level.
- **A crooked approach** (more than 2° off square) may miss a real stop
  line, and that also shifts the count.
- **Lens correction bends lines.** In software tests, the lens-distortion
  correction bent perfectly level synthetic lines past 2°. Real frames
  start out curved and are straightened, so this is probably a test
  artifact, but lines near the image edges are worth checking on the mat.

## The open question

The robot traded false intersections for missed ones. Which is the
smaller risk on the day, and how would you measure it?
