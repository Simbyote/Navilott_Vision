# detect-range

Built as `make routine-detect-range` (`src/routines/detect_range.py`).

```
Name:            detect-range
Date:            2026-10-08

1. QUESTION      From how far before the stop line does the robot read the traffic light (or the
                 stop sign) right?
2. WHY           D3 and D2 have never been checked on the robot; navigation acts on the votes,
                 and a red read as green runs the light
3. ONE TRIAL     The robot parked at one gap before the stop line; the tester sets one state
                 (light red / yellow / green / off; sign there / taken away); a ~2 s look
4. GROUND TRUTH  The state the tester set; the gap taped from the bumper to the line's near edge
5. ROBOT REPORTS The vote at the end of the look (drive_state, stop_sign), the share of frames
                 that saw the right thing, the frames that saw something that isn't there
6. CONDITIONS    The light or sign posted as on the course; the room lit as on the day
7. TRIALS        Gaps x states: 0, 10, 20, 30, 45 cm x 4 light states = 20 (2 sign states = 10)
8. PASS          Every trial within 20 cm reads right; no wrong read at any gap (first guesses:
                 the course's light and sign positions are TBD)
9. OUTCOMES      Pass: D3 / D2 verified to the range shown. Fail: the saved frame of the trial
                 that failed; HSV ranges (calibrate_lamps.md) or the sign gates
10. SAFETY       Nothing drives
```
