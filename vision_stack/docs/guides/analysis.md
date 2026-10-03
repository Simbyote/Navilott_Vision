# Interpreting Runs (src/analysis)

pytest and the linkers collect data; the modules in `src/analysis/` interpret it. Each reads a CSV or JSON a run already wrote and produces numbers, a figure and a JSON summary next to its input. None of them touch the camera, so they run on the Pi or on a laptop with a copied run folder.

## Requirements

- The project's virtual environment
- `matplotlib` for figures (optional). Without it every tool still prints its numbers and writes its JSON.

## Usage

From `vision_stack/`, environment active:

```
python3 -m src.analysis.<tool>                     newest run that has the right CSV
python3 -m src.analysis.<tool> <folder or csv>     a specific run
python3 -m src.analysis.<tool> --help              options
```

A folder is searched for the tool's CSV names, directly inside it first, then in subfolders, so an `artifacts/<YYYYMMDD_HHMMSS>` hardware run works as-is. With no argument, the newest matching file under `runs/` and `artifacts/` is used.

## Tools

| Tool | Reads | Writes | Look at |
| --- | --- | --- | --- |
| `stage_timing` | `stage_timing.csv`, `stages.csv`, `p3.csv` | `timing_budget.png`, `timing_per_frame.png` | Which stage dominates, and how much of the loop is outside the pipeline |
| `jitter` | `frames.csv`, `stage_timing.csv`, `p3.csv`, `stages.csv` | `jitter.png`, `jitter.json` | p99 and max interval, over-budget streaks, whether spikes are periodic |
| `stability` | `p3.csv`, `lane_offset_timing.csv`, `stages.csv` | `stability.png`, `stability.json` | Offset std (noise floor), transitions and flickers per 100 frames |
| `offset_accuracy` | one `p3.csv` per measured position | `offset_accuracy.png`, `offset_accuracy.json` | Bias per position, fit slope and intercept, the PASS/FAIL verdict |
| `gate_rejections` | `geometry_timing.csv`, `color_timing.csv` | `gate_rejections.png`, `gate_rejections.json` | The gate with the largest share, per detector |
| `state_timeline` | `p3.csv` | `state_timeline.png`, `state_timeline.json` | Hold and stale dwell times, stop latching, frequent transitions |
| `soak` | `system.csv`, `soak_frames.csv` from `test_soak` | `soak.png`, `soak.json` | Throttle flags, RSS trend in MB/min, loop time per minute against temperature |
| `detection_range` | one `p3.csv` or `fusion_timing.csv` per measured distance | `detection_range.png`, `detection_range.json` | Reliable range per target, and where detection falls off |
| `compare_runs` | two runs' JSON summaries | `compare.csv` | Every value that moved by 10% or more between two runs |
| `nav_run` | `nav.csv` from `navigation_linker` / `intersection_linker` | `nav_run.png`, `nav_run.json` | The findings list; each intersection's turn end and the 2 s after it; weaving; wheel imbalance |

## Recording for each tool

**Stage timing, jitter, gate rejections:** any normal run. `pytest --hardware` produces all of their inputs; `jitter` also reads a bench run's `stages.csv`.

**Stability:** robot parked, centered in the lane, nothing moving, about 30 s:

```
python3 -m src.phase3_linker --camera              (or the p3.csv of any still run)
python3 -m src.analysis.stability runs/<run>
```

The offset's std is the measurement noise floor. Transitions on a still scene are all flicker.

**Offset accuracy:** one still run per position. Measure each position from lane center with a ruler, using the same sign convention as `lane_offset_cm`. Five positions from -4 to +4 cm is enough. List them in a manifest:

```
true_cm,run
-4,runs/20261001_140000
-2,runs/20261001_140130
0,runs/20261001_140300
2,runs/20261001_140430
4,runs/20261001_140600
```

```
python3 -m src.analysis.offset_accuracy positions.csv
```

Needs `lane_offset_cm`, which Phase 3 writes once its cm scale is configured. Before that, pass `--cm-per-unit` with a hand-measured scale. The exit status is 0 for PASS and 2 for FAIL. A negative fit slope means the offset's sign is flipped, which fails on its own.

**Soak:** robot on the course or the bench, camera live, 10–30 minutes:

```
pytest --hardware --soak-minutes=15 src/tests/test_soak.py
python3 -m src.analysis.soak artifacts/<YYYYMMDD_HHMMSS>
```

The test runs Phases 2–3 with no display or video and samples temperature, CPU clock, throttle flags and memory once a second on the same clock as the frames. It's skipped unless `--soak-minutes` is given, so a normal `pytest --hardware` run isn't held up. Ctrl-C ends it early and still writes everything. Leave the case closed the way it will be on demo day; an open case runs cooler. A memory verdict needs at least 5 minutes after the first minute of warm-up; shorter runs report the trend but don't call it a leak. Under `--replay` the frames loop until time is up: heat and memory are real, camera timing isn't.

**Navigation runs:** any `navigation_linker` run, or one maneuver's folder of an `intersection_linker` run:

```
python3 -m src.analysis.nav_run runs/nav_20261003_101500
python3 -m src.analysis.nav_run runs/intersection_20261003_101500/left
```

It reports, from `nav.csv`:
- **Rules:** time and episodes per deciding rule, the commonest changes between rules, braking and why.
- **Lane keeping:** time on vision / hold / stale; the offset's mean (a bias), spread and p95; **weaving**, steering sign changes per second (past a 0.02 deadband, only within unbroken lane keeping); time at full steering.
- **Each intersection** (one unbroken run of intersection, stop-sign or traffic-light frames): its route step, stage times, time held, how the turn ended, the turn angle by the rule and by the gyro (net of `--gyro-bias`, default `config.GYRO_BIAS_DPS`), and **the 2 s after it**: offset, weaving, how long until both lane lines are back on vision, and a **veer** flag when the offset passes 1.5x the p95 of normal lane keeping (vision frames outside these windows).
- **Wheel balance:** at equal commanded duty, how much faster one wheel turns. + = left faster, which drifts the robot right.
- **Latency:** frame to motor command, and frames that came more than 1.5 budgets after the last.

The findings list says, in words, what passed a threshold (`nav_run.py`'s constants). The thresholds are starting values: after a few good runs, set them to what normal looks like.

**Detection range:** robot still, target placed straight ahead at measured distances, one short run per distance (about 10 s each). Measure from the same point on the robot every time, such as the lens. List the runs in a manifest:

```
distance_cm,run
20,runs/20261001_150000
40,runs/20261001_150130
60,runs/20261001_150300
80,runs/20261001_150430
100,runs/20261001_150600
```

```
python3 -m src.analysis.detection_range distances.csv
```

The reliable range is the farthest distance out to which every distance, from the nearest, is detected in at least 90% of frames (`--threshold`). Targets never detected in any run are left out unless named with `--targets`. Repeat the sweep under the lighting of the demo room; detection range changes with light.

**Compare runs:** two hardware runs, before and after a change:

```
python3 -m src.analysis.compare_runs artifacts/<before> artifacts/<after>
```

Prints each side's git commit from `run_meta.json`. It flags size, not direction: a lower `stage_ms.p95` is good, a lower `frames_with_lane` isn't.

## Cautions

- Frame IDs line up across tests only under `--replay`. On live hardware runs each test captures its own frames.
- `--replay` makes capture a disk read. Its capture and loop times aren't camera timings; use a live run for those.
- The first 5 frames are skipped by default (`--skip`), for auto-exposure settling and first-call allocation.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named src` | Not in `vision_stack/`, or environment not active |
| `no ... found` | The run doesn't contain that tool's CSV; check the Reads column |
| `no lane_offset_cm` | Configure Phase 3's cm scale, or pass `--cm-per-unit` |
| `matplotlib not installed` | Figures skipped; copy the run folder to a laptop and rerun there |
