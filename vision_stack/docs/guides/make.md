# The Makefile: every command in one place

> `make` lists every command the robot has, grouped by what it's for. Each one is a short name for the
> full `python3 -m src.…` command, so nobody has to remember the flags.

**File:** `vision_stack/Makefile` · **Tests:** `test_makefile.py`

---

## Using it

From `vision_stack/`:

```
make                        the list, grouped: race day, linkers, diagnostics, analysis, tests, calibration
make session TIME="2026-10-05 14:30"   on the Pi, every boot: clock, pigpiod, a venv shell
make run                    the course
make nav-dry                the whole chain on the bench, motors off
make nav-report             the newest navigation run's report
```

Commands run on the venv `setup.mk` creates (`~/.venv/navilott`, the same `VENV_DIR`), so it needn't
be activated; without one they use `python3`. Every command is printed before it runs, so you can see
(and copy) exactly what `make` did.

## How it fits with setup.mk and session.mk

The repository root holds two more makefiles, for the Pi itself:

| File | Run | What it's for |
|---|---|---|
| `setup.mk` | once, on a fresh Pi: `make -f setup.mk setup` | apt packages, the camera stack, I2C, pigpio as a service, the venv |
| `session.mk` | every boot: `make -f session.mk session TIME="..."` | set the clock (the Pi has none), start pigpiod, open a venv shell |
| `vision_stack/Makefile` | any time | run the robot's code: the course, linkers, diagnostics, analysis, tests |

This Makefile doesn't repeat them: `make session` and `make pigpiod` here call `session.mk`'s
`session` and `pigpiod-start`, so a change to either is made in one place.

## Options

Options are make variables, written after the target: `make <target> NAME=value`.

| Option | Used by | What it does |
|---|---|---|
| `VIDEO=clip.mp4`, `FRAMES=dir` | the linkers | replay instead of the camera. A replay never drives the motors and doesn't need pigpiod. |
| `ROUTE=my_route.json` | `run`, `navigate`, `nav-dry`, `diag-*` | another course plan (default `route.json`) |
| `MAX_RUN_S=60` | the runs | cap the run's length |
| `VIEWS=lanegeo,traffic` | `phase2` | extra debug views: `lanegeo`, `stop`, `traffic`, `stopline` |
| `MANEUVER=left` | `intersection*` | `straight`, `left`, `right` or `all` (default) |
| `RUN=runs/<folder>` | the analysis targets, `render` | which run to read; default the newest |
| `NAV=runs/nav_<time>` | `pi-load` | the run the recording watched, so only its time is judged |
| `BASE=… NEW=…` | `compare` | the two runs to compare |
| `MINUTES=30` | `soak` | soak length (default 15) |
| `HW_FRAMES=600` | `test-hw` | frames per hardware test (default 100) |
| `LAMPS="red=… green=…"` | `calib-lamps`, `sweep-lamps` | each lamp's recording; for the sweep, `color=run:A-B` labels frames A to B |
| `ARGS="…"` | every target | anything else, passed straight to the command |

`ARGS` covers every flag the Makefile doesn't name, e.g. `make phase3 ARGS="--verbose"` or
`make calib-lamps LAMPS="green=runs/lamp_green" ARGS=--write`.

## Common sessions

```
# a bench run with diagnostics, then its graphs
make diag-nav MAX_RUN_S=60
make pi-load NAV=$(ls -dt runs/nav_*/ | head -1)
make nav-report

# a perception problem on recorded footage
make phase2 VIDEO=runs/clip.mp4 VIEWS=lanegeo,stopline

# before and after a change
make compare BASE=runs/nav_20261003_101500 NEW=runs/nav_20261005_140200
```

## Notes

- Targets that use the camera start `pigpiod` first if it isn't running (`sudo`), because the encoders,
  motors and start button need it. `phase2` and replays don't.
- `compare`, `calib-lamps`, `sweep-lamps` and `render` refuse to run without their argument, and say what it is.
- `make newest` shows the newest run folder of each kind; `make clean` removes Python caches only,
  never `runs/` or `artifacts/`.
- Adding a command: add a target with a `## description` comment after the colon and add it to
  `.PHONY`. `test_makefile.py` then checks its module exists and takes every flag it passes.
