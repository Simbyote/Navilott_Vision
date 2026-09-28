# Test Procedure

When to run which tests. How to run them is in `pytest.md`; what each covers is in `tests.md`; interpreting results is in `analysis.md`.

Run pytest from `vision_stack/` and all other python files from `Navilott_Vision/`, with the environment active.
On the Pi, set the clock first (`pytest.md`, step 1).

## Tiers

| Tier | When | Run |
| --- | --- | --- |
| 0. Software | After every code change | `pytest` |
| 1. Health | Start of every Pi session | `pytest --hardware src/tests/test_system_monitor.py src/tests/test_capture.py` |
| 2. Characterization | After changing tuning or a stage | See below |
| 3. Scenarios | When tuning settles; before reviews | See below |
| 4. Soak | Before the demo; after major changes | See below |
| Calibration | After calibrating, or if the mount or focus changes | `test_calibration.md` |

Tier 0 covers every software test, including the interpreter and debug-view tests. They need no separate runs.

## Live or replay

- Timing (Tier 1, `test_stage_timing`, soak): live camera. Under `--replay`, capture is a disk read.
- Detection logic: `--replay=src/tests/data/frames`. Same frames every run, so differences come from the code.

## Tier 0: Software

Pass: no software hiccups

```
pytest
```

## Tier 1: Health

Pass: no failures, no `camera unavailable` skip. Note the effective FPS in `test_capture/summary.json`.

```
pytest --hardware src/tests/test_system_monitor.py src/tests/test_capture.py
```

## Tier 2: Characterization

```

pytest --hardware --replay=src/tests/data/frames
pytest --hardware src/tests/test_stage_timing.py

```

Then, on each artifact folder:

```

python3 -m src.analysis.stage_timing artifacts/<run>
python3 -m src.analysis.jitter artifacts/<run>
python3 -m src.analysis.gate_rejections artifacts/<run>
python3 -m src.analysis.compare_runs artifacts/<baseline> artifacts/<run>

```

The baseline is the last run you accepted. Keep its folder.

## Tier 3: Scenarios

All use `phase3_linker`. Name each run with `--out`. `--cm-per-px` is the measured ground scale; without it `lane_offset_cm` is empty.

**Still scene** (robot parked and centered, nothing moving):

```

python3 -m src.phase3_linker --camera --limit=400 --cm-per-px=<scale> --out=runs/still
python3 -m src.analysis.stability runs/still
python3 -m src.analysis.state_timeline runs/still

```

**Offset accuracy** (one run per measured position from lane center, e.g. -4, -2, 0, 2, 4 cm):

```

python3 -m src.phase3_linker --camera --limit=200 --cm-per-px=<scale> --out=runs/pos_-4
...
python3 -m src.analysis.offset_accuracy positions.csv

```

`positions.csv` has columns `true_cm,run`. Pass: verdict PASS.

**Detection range** (one run per measured distance, target straight ahead, measured from the lens):

```

python3 -m src.phase3_linker --camera --limit=200 --out=runs/dist_40
...
python3 -m src.analysis.detection_range distances.csv

```

`distances.csv` has columns `distance_cm,run`. Record under the demo room's lighting.

## Tier 4: Soak

Case closed, as on demo day. Ties up the camera; run it last.

```

pytest --hardware --soak-minutes=15 src/tests/test_soak.py
python3 -m src.analysis.soak artifacts/<run>

```

Pass: no throttling, no suspected leak, no slowdown in the findings.

## First session

1. Pull, set the clock, run `pytest` on the Pi.
2. Tier 1.
3. Record the dataset on the course: `pytest --hardware --record --frames=300 -k capture`
4. Tier 2. Keep this folder as the first baseline.
5. Still-scene run (Tier 3).
6. Soak, if time allows.

Sweeps wait until the ground scale is measured.

## Run log

One line per session in `runs/LOG.md`:

```

YYYY-MM-DD | commit | what was run | folder | result in one sentence

```

`run_meta.json` records the commit; the log records why the run was made.
