# recover-offset

Built as `make routine-recover-offset` (`src/routines/recover_offset.py`).

```
Name:            recover-offset
Date:            2026-10-08

1. QUESTION      From how far off the lane's center does the robot steer back, and how close to
                 the center does it hold a straight?
2. WHY           Ignacio's IDR navigation tests (slide 10): maximum recoverable offset; mean
                 distance from the lane's center on a straight run
3. ONE TRIAL     The robot placed at an offset on a long straight, pointing along the lane;
                 it drives until it has held the center 1 s (or 6 s). A start of 0 is the
                 straight run: 5 s of lane keeping
4. GROUND TRUTH  Ruler: lane center to the robot's centerline, before and after
5. ROBOT REPORTS Its own lane_offset_cm (trusted once lane-offset passes): start, time to the
                 center, overshoot, mean and max on the straight, time off vision
6. CONDITIONS    A straight of 1.5 m or more; the scale lane-offset measured (cm_per_px)
7. TRIALS        9: 0, +-2, +-3, +-4, +-5 cm
8. PASS          Every start up to 3 cm recovered (ends within 1.5 cm by the ruler); the
                 straight run never more than 1.5 cm off (first guesses; 1.5 cm is the
                 clearance of an 11 cm robot in a 14 cm lane)
9. OUTCOMES      The maximum recoverable offset; fail: the attempt's run folder (make render)
10. SAFETY       Motors on: someone walks beside it; Ctrl-C stops it
```
