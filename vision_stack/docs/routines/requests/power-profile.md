# power-profile

Built as `make routine-power-profile` (`src/routines/power_profile.py`).

```
Name:            power-profile
Date:            2026-10-08

1. QUESTION      How far does the pack sag, and does the Pi stay unthrottled, under each part of
                 the robot running?
2. WHY           R4 (CPU under load); the battery's margin to its warning level over a run;
                 FDR evidence on power
3. ONE TRIAL     One stage for stage_s (60 s): rest, camera, pipeline (motors off), motors (wheels
                 up, base duty), full (wheels up, pipeline and wheels together)
4. GROUND TRUTH  None by hand: the ADS1115's raw pack volts, /proc/stat CPU, the SoC temperature,
                 vcgencmd's under-voltage and throttle flags, every 0.5 s
5. ROBOT REPORTS The same readings; per stage: mean and lowest volts, sag against rest, drain in
                 mV/min, CPU, hottest, flags
6. CONDITIONS    Pack charged; robot cool; wheels up for motors and full; camera at the course for
                 pipeline and full
7. TRIALS        5 (one per stage); redo repeats a stage
8. PASS          Lowest pack volts under any load at or above VOLTAGE_WARNING (10.5 V); no stage
                 with the Pi's under-voltage; no stage throttled
9. OUTCOMES      Pass: the pack and supply hold under the full load. Fail: low volts -> charge
                 policy or the warning level; under-voltage -> the Pi's 5 V regulator under motor
                 load; throttling -> heat or the supply
10. LIMIT        Voltage only: watts need a current sensor (INA219 / INA226 on I2C, 0x40)
```
