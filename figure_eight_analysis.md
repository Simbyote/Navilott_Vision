# Analysing two figure-eight runs

What to run, in what order, and what each command tests, once two figure-eight runs with frames are on the Pi. Every command runs from `~/Navilott_Vision/vision_stack`.

Below, the two runs are called **A** (the earlier) and **B** (the later):

```bash
A=runs/routine_figure-eight_<time A>/attempt_01
B=runs/routine_figure-eight_<time B>/attempt_01
```

Each attempt folder holds `nav.csv` (every frame), `p3.csv`, `report.json`, `summary.txt`, `intersections.csv`, `laps.csv`, `slips.csv`, `records.pkl` and `frames/`. The routine saves frames by default (`record=1`). Recording a run:

```bash
make routine-figure-eight ARGS="--set minutes=10 --tester <name>"
```

---

## 1. Did the run work? (each run)

| Command | What it tests | Look at |
| --- | --- | --- |
| `make routine-report RUN=$A/..` | The routine's own verdict: each criterion against its limit, every trial, and `nav-report` on each attempt | The criteria table. Your route's straights inflate "turns not ended on the gyro" and "turns off by over 20°": subtract the straights |
| `make fig8-report RUN=$A` | Every turn's heading curve, where each turn ended against its 90°, laps, slips (the chassis not following its wheels), touches; now also motor health | Turns ended on the time limit (hollow points), turns past ±20°, the same corner failing every lap |
| `make fig8-report RUN=$B BASE=$A` | The same for B, with A's turn errors beside it | Whether the eased left turn and 20 kHz PWM changed where the lefts land |
| `make motor-health RUN=$A` (and `$B`) | Each wheel's counts/s per unit duty in 5 s windows against how it started; a wheel driven but not turning | A drop of more than 15% (the failing motor showed 25%); `MOTORS` lines in `summary.txt` say the same |

## 2. How well did it drive? (each run)

| Command | What it tests | Look at |
| --- | --- | --- |
| `make nav-report RUN=$A` | Which rule decided each frame; lane keeping (offset mean and spread, weaving, steering at its limit, stale lane); each intersection and the 2 s after it (veer, lines back); wheel balance; latency | The offset mean: a lean shows as a bias (re-run `make calib-lane` if it's past the threshold); veers after crossings; the wheel imbalance |
| `make timeline RUN=$A` | How long each rule, stage and state lasted, and what changed into what | Unexpected changes (a stop with no sign, a light vote flickering) |
| `make render RUN=$A`, then `make light-report RUN=$A` | `render` makes `nav.avi` and `nav_video.csv` (needed by `light-report`); `light-report` shows the traffic light frame by frame: each reading, the vote, when it braked | Red read as green or nothing, a vote that let go early, stray reds |

## 3. Did it keep its timing? (each run)

| Command | What it tests | Look at |
| --- | --- | --- |
| `make loop-budget RUN=$A` | Where each frame's 50 ms goes: capture, Phases 2–4, time outside the timers, by lane mode | Which phase is over budget when driving |
| `make timing RUN=$A` | Per-stage times against the 50 ms budget | The largest stages (preprocess, geometry, colour) |
| `make jitter RUN=$A` | Frame-interval tails, over-budget streaks, periodic spikes | Streaks of late frames, a spike every N seconds |

## 4. Are the detectors right? (needs labels: the verification work)

Label each run once, then score it. The labels stay with the run, so it can be re-scored after every tuning change.

```bash
make label RUN=$A ARGS="--conditions room,glare,dim"     # window: needs a display (monitor, VNC, or a laptop)
make label RUN=$A ARGS="--template"                      # or a labels.csv to fill in by hand
make eval  RUN=$A
make eval  RUN=$B
```

| Command | What it tests | Look at |
| --- | --- | --- |
| `make label RUN=$A` | Writes `$A/labels.csv`: for ranges of frames, what was really in view: stop sign yes/no, light off/red/yellow/green, stop-line and obstacle distance from the tape, lane offset, range to the sign or light, a condition tag | Label at least 20 frames with each thing present, in each condition you want to claim; also label stretches with nothing there, for false alarms |
| `make eval RUN=$A` | Replays every frame through the robot's pipeline under the current `config.py` and scores each detector against the labels, frame by frame and as voted: recall and precision with 95% intervals, false alarms, the light's confusion matrix, distance error against the tape, all by range and by condition. Writes `eval.json`, `eval.png` | Recall by range (where detection ends), false alarms on empty stretches, the light matrix's off-diagonal (a wrong colour is the hazard), any condition where recall falls |
| `make sign-check ARGS="--frames $A/frames"` | For stop-sign misses: which gate refused the sign in each frame, and whether a larger red blob hid it | The gate behind the misses `eval` found |

**Before and after a tuning change**, on the same frames:

```bash
cp $A/eval.json $A/eval_before.json
# change config.py
make eval RUN=$A BASE=$A/eval_before.json
```

It prints each headline number before → after. Same frames, same labels, so the change is the tuning, not the day's lighting.

## 5. A against B

| Command | What it tests | Look at |
| --- | --- | --- |
| `make fig8-report RUN=$B BASE=$A` | Turn errors side by side, by position in the lap; time-limit turns and touches, A → B | Fewer time-limit lefts with the 7.3 s limit, the same landing |
| `make compare BASE=$A NEW=$B` | Every number in the two runs' JSON summaries that moved past a threshold, largest first; `compare.csv` | Frame rate, latency, braking, lane-keeping numbers |
| `make eval RUN=$B BASE=$A/eval.json` | Detector scores B against A. Different frames, so it's a comparison of runs, not of tuning (the report notes the labelled sets differ) | Whether a detector held up on both days, e.g. the stop sign under different light |

---

## Suggested order

1. `routine-report` and `fig8-report` on each run: did it work?
2. `motor-health` on each: are both motors healthy?
3. `nav-report` on each: is lane keeping centred (a bias means `calib-lane`)?
4. `label`, then `eval` on each: the detector numbers for ER.
5. `fig8-report RUN=$B BASE=$A`, `compare BASE=$A NEW=$B`: what changed between the runs.
6. Timing (`loop-budget`, `timing`, `jitter`) only if a run's frame rate dropped.

Every command prints its findings and writes its JSON and PNG next to the run. Most take `ARGS="--out <folder>"` to put them elsewhere (`routine-report` doesn't).
