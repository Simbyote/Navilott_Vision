# Routines

> Prompted hardware tests: one question about the robot, answered over repeated trials and judged PASS or FAIL against criteria agreed up front. Whoever runs one follows the console; nobody needs to know which linker or flags are behind it.

**Code:** `src/routines/` (`harness.py` the shared flow, one module per routine, `ROUTINES` in `__init__.py`) · **Tests:** `src/tests/test_routines.py` · **Requests:** `docs/routines/request_card.md`

---

## 1. Running one

From `vision_stack/`:

```
make routines                                  # what there is
make routine-tape-check                        # run one
make routine-tape-check ARGS="--trials 10 --notes 'blinds open'"
```

| Option | Meaning |
|---|---|
| `--trials N` | more or fewer trials than the routine's default |
| `--tester NAME` | your name (otherwise it asks) |
| `--notes TEXT` | anything worth keeping about this run: lighting, mat, what changed |
| `--out DIR` | where the folder goes (default `runs/routine_<name>_<time>/`) |

The flow:
1. It checks what the routine needs is running. A routine that drives says `make pigpiod` if the GPIO daemon isn't.
2. It shows the question, the requirement it verifies, and what to set up.
3. It records the conditions: time, commit, host, your name, the Pi's temperature, clock and throttling, and the battery's volts.
4. Each trial prompts you through. Numbers you type are checked: `12,5` and `12.5` both work, and anything outside the stated range is asked again. After each trial:
   - **Enter** keeps it;
   - **r** redoes it (you bumped the robot);
   - **d** discards it;
   - **q** keeps it and stops.
5. At the end it judges the kept trials, prints the summary, and writes the folder.

**q at any prompt, or Ctrl-C, stops cleanly.** The routine releases the hardware (a driving routine brakes), and the folder still holds every trial done, marked INCOMPLETE.

### The verdict

| Verdict | Means |
|---|---|
| **PASS** | every trial planned was kept, and every criterion holds |
| **FAIL** | the trials are complete and at least one criterion doesn't hold |
| **INCOMPLETE** | stopped early or stopped by an error: the criteria are shown, but don't count |
| **RECORDED** | a characterization with no pass criteria: the numbers are the result |

`make routine-...` exits 0 for PASS and RECORDED and 1 otherwise, so routines can be chained in a script.

### The folder

| File | What |
|---|---|
| `summary.txt` | the verdict, each criterion with its value and limit, every trial, the conditions at start and end |
| `trials.csv` | one row per kept trial: trial number, seconds since the start, the routine's fields |
| `results.json` | all of it, plus the routine's own extra state |

Results are compared run to run by the commit and conditions recorded in each. A run with uncommitted changes says so.

---

## 2. The routines

| Routine | Question | Verifies | Needs |
|---|---|---|---|
| `tape-check` | How much do this tester's tape readings of one fixed distance vary? | the hand measurement every accuracy routine relies on | nothing |

**`tape-check`.** Measure one fixed distance five times, taking the tape away in between. PASS: a spread (max − min) of 0.5 cm or less, the tightest tolerance in `requirements.md`. Run it once per tester before any accuracy routine: a routine can't judge the robot more finely than the hand measurement it's compared with. It also rehearses the prompts with no hardware.

---

## 3. Asking for a routine

Fill a **test request card** (`docs/routines/request_card.md`): the question, why it matters, one trial, how a person measures the real answer, what the robot reports to compare, the conditions, how many trials, and the pass criteria in numbers. A card is ready when those have answers. Blank fields are questions to settle together.

Good questions come from three places:
- **`requirements.md`'s unverified rows:** P3 lane offset, D4 stop-line distance, R4 CPU and memory, ...
- **demo-day risks;**
- **the design review.**

A question that serves none of them goes on the after-demo list.

Ideas to start from:

| Area | Routine idea | Ground truth |
|---|---|---|
| Navigation | stopping distance at a stop line | tape from the line's near edge to the bumper |
| | lane centering at 0 / ±2 / ±4 cm (P3's procedure, prompted) | ruler from the lane center to the robot's centerline |
| | straight-line drift over 1 m | sideways offset from a taped line |
| | turn accuracy at an intersection (with `intersection`'s PASS / CHECK) | heading against the mat's grid; exit position in the lane |
| | detection range: traffic light and stop sign | marked distances on the mat |
| | course repeatability over N full runs | completion, time, interventions |
| Power | voltage under load stages: idle, camera, the pipeline with motors off, wheels-up driving, a full run | the routine's own volts, temperature, CPU and throttle readings |

**Power has one limit:** the robot measures the battery's voltage, not its current. Voltage gives the sag under each load and the time to the warning level. Watts and energy need a current sensor, e.g. an INA219 or INA226 on the existing I2C bus (address 0x40, free; the bus has room).

---

## 4. Writing a routine (for developers)

A routine is a `Routine` subclass in its own module under `src/routines/`, added to `ROUTINES`:

```python
from src.routines.harness import Routine, criterion, stats

class StopDistance(Routine):
    name = "stop-distance"                       # make routine-stop-distance
    title = "Stopping distance at a stop line"
    question = "How far before a stop line does the robot stop, and how consistently?"
    requirement = "D4"
    trials = 10
    fields = ("gap_cm", "reported_cm", "battery_v")
    needs = ("pigpiod",)
    instructions = "Place the robot on the start mark 60 cm before the stop line, centered."

    def setup(self, ctx):        # open hardware once; ctx.options, ctx.state are free to use
        ...
    def trial(self, ctx, i):     # prompt, run, measure; return a row with the fields
        ctx.console.wait("Robot on the start mark? Enter to drive")
        ...
        return {"gap_cm": ctx.console.ask_number("Gap from the line to the bumper", lo=-20, hi=60, unit="cm"), ...}
    def teardown(self, ctx):     # always runs, even after q, Ctrl-C or an error: brake, release
        ...
    def judge(self, rows):       # the card's criteria
        s = stats(r["gap_cm"] for r in rows)
        return [criterion("mean gap", s["mean"], "within", (2, 6), "cm"),
                criterion("spread", s["max"] - s["min"], "<=", 2, "cm")]
```

Building from a card:
- **One routine per card;** its name is the card's name.
- **Card §3, the trial:** each step is a `console.wait()` or a run of existing code (a linker's run function, the sensors, the battery), never a subprocess the tester has to watch.
- **§4, ground truth:** an `ask_number()` with the range a real answer can have.
- **§5, what the robot reports:** read it in the trial and put it in the row beside the hand measurement.
- **§8, pass criteria:** become `judge()`. A card without numbers yet is a characterization; leave `judge()` returning `[]`.
- **`needs`:** list what must be running, which `check_needs` checks before the first prompt. Add a new need to `harness.NEEDS` with the instruction to fix it.
- **Tests:** script the tester with a fake console, as `test_routines.py` does.
