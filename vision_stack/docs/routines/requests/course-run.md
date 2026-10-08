# course-run

Built as `make routine-course-run` (`src/routines/course_run.py`).

```
Name:            course-run
Date:            2026-10-08

1. QUESTION      Does the robot complete the course, in its lane and obeying the lights and
                 signs, and how fast?
2. WHY           D2: Senior Design Day is this, once, in front of judges
3. ONE TRIAL     The whole course from the route file: start button, countdown, motors on,
                 to the finish
4. GROUND TRUTH  The tester: completed as planned (y/n), a phone stopwatch from GO to the
                 finish, touches, stop signs / red lights run
5. ROBOT REPORTS How the run ended, its own run time (the display's), route steps reached,
                 stop sign holds and red waits, time off vision, contract brakes, frame rate,
                 the pack at start and end
6. CONDITIONS    The course as on the day; a charged pack
7. TRIALS        3 runs
8. PASS          Each run: completed (tester and robot agree); no touches; no stop sign or red
                 light run. Times reported: best and mean, not judged
9. OUTCOMES      Pass: ready for the day. Fail: the attempt's run folder (make render) at the
                 step it failed
10. SAFETY       Motors on: walk beside it; Ctrl-C stops it
```
