# frame-budget

Built as `make routine-frame-budget` (`src/routines/frame_budget.py`).

```
Name:            frame-budget
Date:            2026-10-08

1. QUESTION      Does the whole chain keep 20 FPS, answer within 50 ms, and fit in the Pi's CPU
                 and memory?
2. WHY           P1, P2, R4; the IDR promised pipeline-timing and CPU tests (slide 7)
3. ONE TRIAL     One 60 s run of the whole chain with navigation deciding, motors off, the robot
                 parked in a lane
4. GROUND TRUTH  None by hand: the clock and /proc are the measurement
5. ROBOT REPORTS Per frame (nav.csv): Phases 2+3 time, arrival-to-motor-command latency, rate;
                 every 0.5 s on a side thread: CPU busy %, this process's memory, temperature;
                 throttle flags at the start and end
6. CONDITIONS    Nothing else running on the Pi; lit as on the course
7. TRIALS        3 runs
8. PASS          Each run: >= 20 FPS (P1); <= 5% of frames over the 50 ms budget ("near 0", a
                 first guess); p95 latency <= 50 ms (P2); CPU mean <= 70% and memory <= 400 MB
                 (R4); ran the whole 60 s
9. OUTCOMES      Pass: P1, P2 (from appsink), R4 verified. Fail: the attempt's nav.csv and
                 summary for the slow stages (P1 is known short: 15 FPS on the IMX290)
10. SAFETY       Nothing drives
```
