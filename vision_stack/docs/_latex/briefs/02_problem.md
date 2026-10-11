# Brief: Problem

**This brief covers only the Problem section: what the robot had to do,
under what constraints, on what course, and how that became checkable
requirements. It does not cover how the robot was built.**

## The section's main idea

Before any design there was a job, a set of constraints and a course, and
the three don't fully agree. The section's work is to turn the PRD's prose
into requirements that can each be checked one way.

## The job (from the PRD)

Navilott must follow lane lines, drive on the right, obey traffic lights
and stop signs at intersections, and reach the end of a course without
help. Obstacle avoidance was a stretch goal in the team's notes; the PRD's
executive summary lists it as a must.

## Constraints (from the PRD and the team's notes)

- A budget of $100.
- A custom PCB that carries the robot's electronics.
- Power at 12 V or less.
- A size limit of about 10 x 11 x 11 cm.
- All processing on board, no network.
- A start button and a 5 s countdown; elapsed time shown in tenths of a
  second; battery state shown to the user.

## The course: PRD against the team's course

| | PRD | Team's course |
|---|---|---|
| Layout | Grid of at least nine 4-way intersections | Six intersections |
| Lights | At four or more intersections | Four traffic lights |
| Stop signs | One, at the exit | At every intersection without a light |
| Surface | Black foam mats, 3/4 inch white tape | Dark mats, white tape; lanes about 14 cm wide |
| Center line | Dashed | Dashed |

The team's practice arena is 250 x 300 cm, 18 mats. The traffic lights on
the course turned out to be bare 5 mm LEDs on a board, not lamps.

## Requirements (vision_stack/docs/requirements.md)

- **Performance:** P1 at least 20 frames per second; P2 a decision within
  50 ms of the frame; P3 lane position within ±2 cm; P4 on board only.
- **Detection:** D1 lane boundaries, D2 stop signs, D3 traffic light color,
  D4 stop line and its distance, D5 intersections.
- **Robustness (internal targets, not PRD):** R1 a dropped frame never
  stops the robot; R2 a lost lane is held at most about 350 ms; R3 no single
  frame flips a light or sign; R4 CPU under 70% and memory under 400 MB;
  R5 the motors stop if commands stop.

Some are straight from the PRD; others are derived, things vision needs so
that a PRD rule can be met.

## Questions the author has to answer here

1. Which constraints actually shaped the design, and which were boxes to tick?
2. Was obstacle avoidance formally dropped, by whom and when?
3. Does each course difference make vision easier or harder?
4. Why these numbers: 20 FPS, 50 ms, ±2 cm?
