# Question: What happens when something breaks mid-run?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how the robot fails safely: what each layer does when
a part stops working. Nothing else about the robot is in scope.**

## The short answer

In layers, each one assuming the one above it might fail. The rule
throughout: stopping is always a safe answer, and a guess dressed up as a
measurement never is.

## The layers, from the camera to the wheels

- **A dropped frame:** skipped. It doesn't use up a frame number, and the
  robot keeps going. A dead camera is different: about 1 s of failed reads
  in a row (20) raises an error and stops the run. (Requirement R1,
  verified in software.)
- **A lost lane:** the lane is held for about 350 ms, then marked stale.
  The robot creeps on its gyro heading, and after 1 s of that it brakes and
  ends the run. (R2.)
- **A flickering detection:** no single frame can flip a light, sign, stop
  line or obstacle decision; 2 of the last 3 must agree. (R3.)
- **A bad command:** every command navigation returns is checked against
  the contract before it reaches the motors:
  - each duty within −1..+1;
  - a driven wheel at least 0.25, where the motors would otherwise stall;
  - a brake carries no duty.

  A command that breaks a rule becomes a brake, with the reason logged.
- **Something in the way:** the obstacle rule is the first rule, above
  everything else. It brakes for anything in the path within 12 cm.
- **A frozen program:** a motor watchdog on its own thread brakes the
  motors if no command has arrived for 0.5 s (R5). It's verified in
  software, with a hardware test written but not yet recorded. The wheels'
  PWM comes from the Pi's hardware, so a late Python thread delays a change
  but never garbles the signal.
- **A tired battery:** below 10.5 V the robot warns; at 9.9 V it ends the
  run. The pack has no battery management board of its own, so this check
  is the only protection.

## The idea underneath

Each layer reports how much to trust what it hands on:
- capture gives a frame or an error;
- estimation labels the lane vision, hold or stale;
- navigation is free to brake.

No layer hides a failure from the next.

## The open question

Which of these has actually been seen working on the robot, and which only
in tests? The watchdog's hardware test, for one, hasn't been recorded yet.
