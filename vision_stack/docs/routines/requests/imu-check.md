# imu-check

Built as `make routine-imu-check` (`src/routines/imu_check.py`).

```
Name:            imu-check
Date:            2026-10-08

1. QUESTION      Is the gyro's bias still the configured one, does a real 90 deg read 90, and
                 what do the motors add?
2. WHY           Every turn ends on the gyro's heading and every hold subtracts GYRO_BIAS_DPS:
                 a moved bias drifts the heading (0.3 deg/s is 18 deg a minute), a scale error
                 ends turns early or late, a wrong IMU_YAW_SIGN turns the wrong way
3. ONE TRIAL     Part 1: 60 s still on the floor. Parts 2-9: one 90 deg hand turn each, lined
                 up on the mat's grid, alternating left and right. Part 10: 30 s wheels up at
                 base duty
4. GROUND TRUTH  The mat's grid: 90 deg between line-ups; at rest, no turn at all
5. ROBOT REPORTS The gyro through the sensor hub: bias, noise, the heading's drift with the
                 configured bias, each turn's heading, read rate, read time, read errors, temp
6. CONDITIONS    Robot on the floor (not a table someone leans on); a box for wheels up
7. TRIALS        10: 1 rest, 8 turns, 1 vibration (--set rest_s=..., vib_s=...)
8. PASS          Bias within 0.2 deg/s of GYRO_BIAS_DPS; rest drift within 2 deg over the
                 minute; every turn within 3 deg of 90 (the wrong way is 180 off); no read
                 errors. Vibration noise reported, not judged
9. OUTCOMES      Pass: the heading config holds. Bias off: the summary says the value to put in
                 config.py. Turns off one way: the left/right scale. Both directions reversed:
                 flip IMU_YAW_SIGN
10. SAFETY       Part 10 runs the motors: wheels off the ground; Ctrl-C stops them
```
