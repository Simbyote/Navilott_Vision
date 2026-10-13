# Question: Why does yellow mean stop on this robot?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how the robot decides to wait or go at a traffic
light. Nothing else about the robot is in scope.**

## The short answer

Because a red LED, seen from some angles, looks yellow to the camera. Until
2026-10-07 the robot stopped only on red. Then a red light whose overexposed
ring read orange, inside yellow's color band, was taken for yellow, and the
robot drove through it. Stopping on yellow too means a red misread as yellow
still stops the robot, and a real yellow turns red anyway. Only green
drives on.

The project's rules (the PRD) allow entering an intersection only on green,
so this rule moved the robot closer to them.

## How the wait works

A traffic light only stops the robot at a stop line; a red light with no
line in front of the robot is ignored.

- **Memory:** at the line, if the voted light state was red or yellow at
  any time in the last 0.5 s (RED_MEMORY_MS), it brakes. The light is a few
  small LEDs, and the detector can miss one for a frame. The rule used to
  need red on the exact frame the line was reached, and one dropped frame
  ran the light.
- **Release:** it keeps braking until the light has read green, or nothing,
  for 0.25 s (RELEASE_MS). One stray frame can't let the robot go on a red.
- **Priorities:** a stop sign at the same line is handled by its own rule
  (brake, wait 2 s, go). While the light rule brakes, the intersection's
  timers don't run, so a turn starts only once the robot is let go.

## The pattern

Each part of this rule came from a specific failure on the mat:
- the memory, from a dropped frame;
- the release time, from a stray frame;
- yellow, from a misread color.

## The open question

Waiting on "nothing seen" for 0.25 s also releases the robot. What if the
light is simply hidden, by a person or a glare, while it's red?
