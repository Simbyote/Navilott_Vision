# figure-eight

Built as `make routine-figure-eight` (`src/routines/figure_eight.py`).

```
Name:            figure-eight
Date:            2026-10-08

1. QUESTION      Does the robot keep driving the course correctly, left and right, lap after lap,
                 for the whole time?
2. WHY           Course repeatability; demo-day endurance: every rule over many intersections, left
                 and right turns side by side, the pack and heat building over minutes
3. ONE TRIAL     One continuous run, motors on: from the stop line of the intersection two blocks
                 share, about to turn left; route left x4, right x4, repeated; ends after the time
4. GROUND TRUTH  The tester counts the times they touched the robot and the laps they saw complete
5. ROBOT REPORTS Per intersection: maneuver, turn end (gyro / time limit), heading against -90/+90,
                 lane back, volts; per lap: time, volts; lane-lost time, contract brakes, frame rate
6. CONDITIONS    Pack charged; the two blocks of track; lights as for a run
7. TRIALS        1 run of 10 minutes (--set minutes=...; --trials N for more runs)
8. PASS          No touches; ran the whole time; robot's laps = laps seen; every finished turn on the
                 gyro and within 20 deg of its maneuver (intersection_linker's tolerance)
9. OUTCOMES      Pass: ready for a long course run. Fail: the intersections.csv row that failed and
                 its frames in the run folder; left vs right means for an asymmetry
10. SAFETY       Motors on: someone near the track; Ctrl-C stops it
```
