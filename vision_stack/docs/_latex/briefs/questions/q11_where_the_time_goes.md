# Question: Why does the robot run at 15 frames a second instead of 20?

**This brief answers one question about Navilott, a small lane-following
robot on a Raspberry Pi Zero 2 W. It covers only the robot's frame rate:
the target, what's been measured, and what's been found and fixed. Nothing
else about the robot is in scope.**

## The target and the gap

The requirement (P1) is at least 20 frames per second, a 50 ms budget per
frame for everything: perception, estimation and navigation. With
navigation running, about 15 FPS has been measured, with frames over budget.
An earlier drive trial (maneuver_linker, 2026-09-30) measured about 19.8 FPS.
Why the two differ isn't settled.

## One frame at a time

The robot processes one frame at a time, in step with the camera. The
camera keeps only its newest frame, so a slow frame means the next one is
the current view, not a backlog: for steering, a skipped frame beats a late
one. The cost is that every millisecond in the loop is a millisecond of
frame rate.

## What the diagnostics found

The Pi's own load is recorded by a separate process that shares nothing
with the robot.

- **Python's GIL:** only one Python thread runs Python at a time. OpenCV's
  C++ code releases the lock, so image processing can overlap other work;
  the Python glue can't.
- **OpenCV's worker threads:** at OpenCV's default of 4 threads, three
  workers each spun about 1,300 times a second waiting for work. Capped at
  2, there's one worker, about 33% busy.
- **The gyro reads:** the sensor thread read the IMU in 7 separate I2C
  transfers, 4.6 ms per reading. A single block read (2026-10-03) cut that
  thread to about 5.6% of the CPU.
- **Garbage collection** (2026-10-12): Python's memory cleanup could land
  inside a frame. Everything alive before the loop is now frozen, so
  cleanups skip it, and each cleanup is timed so a slow frame can say
  whether it was the cause.
- **New work costs time too.** Obstacle detection builds its masks on a
  quarter of the pixels because the loop is already at its budget.

## The design question underneath

One board was chosen over a Pi plus a microcontroller, for one codebase and
no link between them. The camera sets the steering rate either way, so a
faster motor loop on a microcontroller would mostly repeat the last command.
But it means Linux's scheduler, Python and the vision share four small
cores.

## The open question

Is 15 FPS a problem on the course, or only against the number? What would
a frame-budget routine run on the day say about where the remaining time
goes?
