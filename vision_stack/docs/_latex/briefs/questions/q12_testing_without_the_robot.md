# Question: How do you test a robot without the robot?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how its software is tested off the robot, and what
that can and can't prove. Nothing else about the robot is in scope.**

## The short answer

By building the software so every stage can run on a laptop:
- **Estimation never imports a hardware driver.** The gyro and wheel
  readings arrive as plain data.
- **Navigation is pure logic:** a packet goes in, a command comes out.
- **The camera can be swapped** for recorded video, image folders, or
  synthetic scenes drawn in code: lane lines, stop lines, signs and lamps at
  known positions.

That makes about 2,700 automated tests possible, run on any machine in a
few minutes.

## Kinds of test

- **Pinning tests** hold a tuned value in place, for example "the stop-line
  tilt limit is 2°". They fail on purpose when the value changes, so every
  change is a decision someone confirms.
- **Behavior tests** feed scripted packets frame by frame and check what
  the robot decides: brake at a red line, cross a green one, count one
  intersection per stop line.
- **Scene tests** run the real image pipeline on synthetic frames with
  known answers.
- **Contract checks:** every navigator must pass the same checks, such as
  slowing then stopping on a lost lane, or never braking on a green light.

## A story: 41 failures that were the tests working

In October, three deliberate changes on the mat (a 2° tilt limit, a new stop
delay, a new turn stage) broke 41 tests at once. Most were pinning and
behavior tests doing their job: the robot had changed, and the tests still
described the old one. Updating them also found a real bug: a logging tool
crashed on every turn, because it didn't know the new stage.

## What testing can't prove

A passing suite says the code does what it was written to do. It doesn't
say the robot drives the course. Synthetic frames are cleaner than real
ones: drawn already undistorted, evenly lit, with perfect tape. One flaky
test turned out to be the frame recorder dropping frames under load, which
was correct behavior that the test hadn't allowed for. For the real robot
there are hardware routines with pass criteria fixed in advance, run by a
person following prompts.

## The open question

Where did a passing suite hide a problem that only showed on the mat?
