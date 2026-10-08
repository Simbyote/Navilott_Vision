# Test Request Card

> One card per question you want answered on the robot. Fill it in, and it becomes a routine you run with `make routine-<name>`: the console prompts every step, asks for your hand measurements, and gives a PASS / FAIL verdict with every trial recorded.

**How to use it:** copy the blank card below into `docs/routines/requests/<short-name>.md`, fill in what you can, and hand it over. Blank fields are fine; they're questions to settle together. A filled example follows the blank one.

A card is ready to build when the question is one sentence, ground truth has a method, and pass criteria have numbers.

---

## Blank card

```
Name:            (short, lowercase-with-dashes: becomes make routine-<name>)
Requested by:
Date:

1. QUESTION (one sentence; one question per card)


2. WHY IT MATTERS (pick any)
   [ ] Closes a requirement in docs/requirements.md:  row ____
   [ ] Demo-day risk:  ____
   [ ] Evidence for the design review (FDR):  ____
   [ ] Other:  ____

3. WHAT THE ROBOT DOES IN ONE TRIAL
   (where it starts, what it runs, when the trial ends; motors on or off)


4. GROUND TRUTH: HOW A PERSON MEASURES THE REAL ANSWER
   Tool (tape, ruler, protractor, stopwatch, phone video...):
   Measured from ____ to ____ :
   Resolution you can read (e.g. 1 mm):

5. WHAT THE ROBOT REPORTS (if anything) TO COMPARE WITH THE GROUND TRUTH
   (e.g. stop_line_cm in nav.csv, heading_deg, battery_v; or nothing:
   the hand measurement is the result)


6. CONDITIONS TO HOLD FIXED (and record)
   Battery:            (e.g. above 11.5 V at the start)
   Course / mat:
   Lighting:
   Speed / duties:
   Anything else:

7. TRIALS
   How many:            (5 minimum; 10 if the result decides something)
   Between trials:      (reset the robot to the mark, wait N s, swap nothing...)

8. PASS CRITERIA (numbers; who agreed them)
   e.g. "mean stop gap 2-6 cm, spread (max - min) under 2 cm"
   Criterion 1:
   Criterion 2:
   Source of the numbers (requirement, navigation's need, a guess to refine):

9. WHAT WOULD YOU DO WITH EACH OUTCOME
   If it passes:
   If it fails:

10. SAFETY / SET-UP NOTES
   (wheels-up, someone at the robot to catch it, space needed...)
```

---

## Filled example: stopping distance at a stop line

```
Name:            stop-distance
Requested by:    (example)
Date:            2026-10-08

1. QUESTION
   How far before a stop line does the robot stop, and how consistently?

2. WHY IT MATTERS
   [x] Closes a requirement: D4 (stop-line distance), on the navigation side
   [x] Demo-day risk: stopping on or past the line at an intersection
   [ ] Evidence for the FDR
   [ ] Other

3. WHAT THE ROBOT DOES IN ONE TRIAL
   Starts centered in the lane at a tape mark 60 cm before an intersection's
   stop line, motors on, drives with lane keeping until navigation brakes
   for the line (stop sign side), then the trial ends.

4. GROUND TRUTH
   Tool: tape measure
   Measured from: the stop line's near edge, to: the front of the bumper,
                  along the lane's center
   Resolution: 1 mm

5. WHAT THE ROBOT REPORTS
   stop_line_cm on the frame it braked (nav.csv), to compare with the tape

6. CONDITIONS
   Battery: above 11.5 V at the start (the routine records it)
   Course / mat: the course mat, intersection 1, stop sign present
   Lighting: room lights on, blinds closed
   Speed / duties: the configured BASE_SPEED
   Anything else: same start mark every trial

7. TRIALS
   How many: 10
   Between trials: carry the robot back to the start mark, wait 5 s

8. PASS CRITERIA
   Criterion 1: mean gap between 2 and 6 cm (never on the line)
   Criterion 2: spread (max - min) at most 2 cm
   Source: navigation's need (stop before the line, room to see the light);
           a first guess to refine after the first run

9. WHAT WOULD YOU DO WITH EACH OUTCOME
   If it passes: keep STOP_DELAY_MS; rerun at a low battery (10.8 V)
   If it fails: mean off -> adjust STOP_DELAY_MS; spread too big ->
                look at stop_line_cm noise and the robot's speed per trial

10. SAFETY / SET-UP NOTES
   Someone stands at the intersection to catch it if it doesn't stop.
```

---

## Before your first accuracy routine

Run `make routine-tape-check` once per tester. You measure one fixed distance five times, taking the tape away in between. Your spread is the finest difference your hand measurements can show. If your readings vary by 5 mm, a routine can't judge the robot to a millimetre. The routine records it under your name.
