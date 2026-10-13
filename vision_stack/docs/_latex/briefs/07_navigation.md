# Brief: Navigation

**This brief covers only the Navigation section: how the robot turns each
frame's record into wheel commands. It does not cover how anything is
detected (Perception) or smoothed (Estimation).**

## The section's main idea

Navigation turns the per-frame packet into a wheel command. The hand-off
between vision and navigation was written down as a contract on 2026-09-30,
before navigation existed, so both sides could be built and tested apart.
Navigation is a short list of six rules in priority order. The first rule with
something to say wins, but every rule sees every frame, so its timers stay
current.

## The contract

- **Every frame:** navigation gets one packet and must return one command.
  It never blocks, sleeps or raises on ordinary input.
- **A command:** a duty per wheel from −1 to +1, or brake.
- **A wheel that drives** gets at least 0.25 duty: below that, the N20
  motors stall.
- **Navigation never reads the camera or the sensors directly;** vision
  never drives the motors.
- **Enforcement:** a command that breaks the rules is replaced with brake
  before it reaches the motors.

## The rules, highest first

1. **Obstacle:** anything in the path whose base is within 12 cm (or
   already below the view): brake, then wait until nothing has been seen for
   1 s before going on.
2. **Stop sign:** at the line, with a sign seen in the last 5 s, it brakes
   until the wheels read stopped (or 1 s passes), holds 2 s, then goes.
3. **Traffic light:** at the line, with red or yellow seen in the last
   0.5 s, it brakes until it has read green (or nothing) for 0.25 s.
   - **Yellow stops too** (since 2026-10-07): the overexposed red LED can
     read yellow, and a real yellow turns red anyway. The PRD allows
     entering an intersection only on green.
4. **Intersection:** takes over from the moment the stop line leaves the
   view. Lane keeping would be pulled off course by the crossing street's
   lines. Its stages:
   - **to the line,** on a gyro heading hold;
   - **turn** (left or right only): until the gyro reads 85°, or a time
     limit if it never does. Left is a wide arc into the far lane, (0.50,
     0.73) as duties; right is a tight arc, (0.50, 0.25). Since 2026-10-12
     the turn asks for the wheel speeds those duties gave at about 11.7 V,
     using the encoders, and finds the duty that gets them on today's
     battery;
   - **exit:** straight until the lane lines are back.
5. **End of course:** a lane lost for more than 1 s at slow speed ends the
   run. That's the finish if the route is done; otherwise it's a safety
   stop.
6. **Lane keeping:** when no rule speaks, it steers by the lane offset.
   - **Base duty** is 0.40, steering at most ±0.40.
   - **Gain** grows with the offset.
   - **When vision is lost,** it steers by the heading turned instead.

## The stop line, from first sight to "reached"

The camera's view of the floor ends about 10 cm ahead of the robot, so the
line leaves view before the robot reaches it. A tracker follows the line:
seen, gone from the bottom of the view (one intersection counted), then
reached. The delay after it leaves view (STOP_DELAY_MS) has changed:

- 1500 ms stopped the robot past the line;
- 500 ms stopped it at the line (2026-10-05);
- 0 on 2026-10-07: the course's new traffic light hangs over the center of
  the intersection, and 500 ms of driving carried the robot under it and
  out of view. Turns then drove 1 s further first (an "advance" stage);
- 500 ms again on 2026-10-12, with the advance set to 0. The commit gives
  no reason, and the code's comment still describes the 0.

## The route

A file, route.json, lists one maneuver per intersection (left, right or
straight) and how the run finishes. Every stop line that passes under the
view is the next intersection. A missed or false stop line shifts every
later turn, which is why the stop-line tilt gate was tightened to 2°.

## Measured robot response (maneuver_linker, 2026-09-30)

- **Speed:** 0.40 duty drives about 1166 encoder counts/s per wheel.
- **Coasting:** keeps turning for 0.15–0.18 s; braking stops sharply.
- **Frame timing:** about 19.8 FPS, frame gaps up to 0.16 s.

## Questions the author has to answer here

1. Who designed which rule? Navigation was built by Mike, Ignacio and
   AI-assisted sessions.
2. Where does the robot's behavior differ from the PRD's rules?
3. Which numbers (delays, duties, frame counts) were tuned on the mat, and
   which are still guesses?
