# lane-offset

Built as `make routine-lane-offset` (`src/routines/lane_offset.py`).

```
Name:            lane-offset
Date:            2026-10-08

1. QUESTION      Does the robot's reported position in the lane match a ruler, to within 2 cm?
2. WHY           P3; lane keeping steers on this number, and recover-offset needs it trusted
3. ONE TRIAL     The robot parked at one offset from the lane's center, pointing along it; a
                 100-frame look
4. GROUND TRUTH  Ruler: lane center (halfway between the lines' centers) to the robot's
                 centerline, + = right of center
5. ROBOT REPORTS The mean filtered lane_offset over the frames on vision, in cm through the
                 ground scale the first (centered) trial measures: 14 cm / lane width in px
6. CONDITIONS    A straight lane, both lines in view, lit as on the course
7. TRIALS        5: 0, +2, -2, +4, -4 cm (requirements.md's P3 procedure)
8. PASS          Every position within 2 cm of the ruler (P3)
9. OUTCOMES      Pass: P3 verified; put the measured cm_per_px in config.py. Fail: the
                 position's saved frame; the lane-centre projection (course.md)
10. SAFETY       Nothing drives
```
