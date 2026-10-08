# stop-distance

Built as `make routine-stop-distance` (`src/routines/stop_distance.py`). The card is `request_card.md`'s filled example; its numbers live in the routine as `GAP_RANGE_CM`, `MAX_SPREAD_CM` and `START_GAP_CM`.

```
Name:            stop-distance
Date:            2026-10-08

1. QUESTION      How far before a stop line does the robot stop, and how consistently?
2. WHY           D4 (stop-line distance), the navigation side; demo-day risk: stopping on or past the line
3. ONE TRIAL     From a start mark 60 cm before an intersection with a stop sign (or a red light),
                 motors on, lane keeping until navigation brakes for the line and the wheels stop
4. GROUND TRUTH  Tape, from the stop line's near edge to the front of the bumper along the lane's
                 center, to 1 mm; negative past the line
5. ROBOT REPORTS The last stop_line_cm before braking; speed at braking; battery volts
6. CONDITIONS    Pack above 11.5 V at the start; the course mat, intersection 1; room lights on,
                 blinds closed; BASE_SPEED; the same start mark every trial
7. TRIALS        10; carry back to the mark between trials
8. PASS          Every trial stopped before the line; mean gap 2-6 cm; spread (max - min) at most 2 cm.
                 Source: navigation's need; first guesses to refine after the first runs
9. OUTCOMES      Pass: keep STOP_DELAY_MS, rerun at a low pack (10.8 V).
                 Fail: mean off -> STOP_DELAY_MS; spread -> stop_line_cm noise, speed per trial
10. SAFETY       Someone at the intersection to catch it
```
