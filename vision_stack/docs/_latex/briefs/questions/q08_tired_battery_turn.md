# Question: How does a turn stay the same on a draining battery?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how the robot's turns use wheel-speed feedback.
Nothing else about the robot is in scope.**

## The problem

The motors are driven by duty: a share of the battery's voltage. The pack
runs from 12.6 V full down to 9.9 V, where the run ends. The same duty turns
the wheels slower as it drains, so a turn tuned on a fresh pack draws a
smaller, slower arc on a tired one, and its slow inner wheel stalls sooner.
Lane keeping corrects itself from the camera every frame; a turn is driven
blind, by fixed commands, so it is where the battery shows.

## The answer: ask for a speed, not a duty (2026-10-12)

Each wheel has an encoder that counts its rotation. Turns now ask for wheel
speeds in counts per second (the speeds the tuned duties gave at about
11.7 V) and work out the duty that gets them today:

    duty = target / gain  +  proportional trim  +  integral trim

- **Feedforward (target / gain):** gain is counts per second per unit duty.
  It starts at 3,000. The 2026-10-12 runs measured about 3,083 per unit plus
  59, over 1,511 steady frames at 11.6–11.8 V.
- **Learned while driving:** on every steady frame of lane keeping, the
  measured speed over the commanded duty nudges the gain 5% of the way. So
  by the time a turn starts, the gain already reflects today's battery and
  floor. It's kept within 0.6–1.4 times the starting value, so a wheel
  spinning up or a lifted robot can't run it away.
- **PI trim:** takes out what's left. The integral only runs once the
  wheel is near its target, so it doesn't wind up while the wheel spins up.
  In simulation, a 15% weaker motor was back on target in about 0.5 s.
- **Without encoders,** it's the feedforward alone, which gives the same
  duties as before.

## Why only turns

A speed loop for all driving was put off until after demo day: it would
change behavior that already worked. This puts feedback only where
open-loop driving was weakest. Lane keeping still steers by duty, with the
camera as its feedback.

## The open question

The trim gains were set in simulation. How would you check on the mat that
a turn at 10.5 V matches one at 12 V?
