# Drive Trial (maneuver_linker)

A scripted run that tests the motors, wheel encoders and IMU working together while the vision pipeline records. The robot:

1. sits still for 2 s to measure the gyro's drift;
2. spins briefly left, then right, to learn which way the IMU counts a turn;
3. drives one leg forward, kept straight by the encoders and the gyro;
4. stops, turns 180° on the gyro, and drives the leg back.

It reports whether the 180° turn passed and what each sensor saw. Vision runs on every frame but only records, it never steers. The video is made after the run, so drawing never slows the robot down.

This is not the Navigation state machine. It checks that the parts Navigation will rely on behave.

**The robot moves on its own.** Read section 3 before the first run on the mat.

## Requirements

- The Pi with the camera, the MPU-6050 (I²C, address 0x68), the TB6612 motor driver and both wheel encoders wired as in `src/peripherals/drive.py`
- The pigpio daemon: `sudo pigpiod` (once per boot)
- The start button and TM1637 display (`src/peripherals/system.py`), or `--no-button`
- Camera not in use by anything else (`phase2_linker`, `phase3_linker`, `rpicam-hello`, a pytest run)
- A charged battery. Low voltage slows the motors, which changes every number this run measures

## 1. Setup

```
cd ~/Navilott_Vision/vision_stack
source .venv/bin/activate
sudo pigpiod
```

Set the Pi's clock first (run folders are named by it):

```
sudo timedatectl set-ntp false
sudo timedatectl set-time "YYYY-MM-DD HH:MM:SS"
```

## 2. Checks before it drives

**a. Motors and encoders, wheels off the ground** (prop the robot up so both wheels spin freely; about 10 s of motor time):

```
python3 -m pytest --hardware src/tests/test_drive.py
```

It spins each wheel forward and back and checks each encoder counts. **Watch the wheels during the first forward spin.** If they turn the way that would drive the robot *backward*, stop and report it: the motor direction needs fixing in `drive.py` before a trial.

**b. Bench run, nothing moves.** This checks the camera, IMU, encoders, recording and video without the motors:

```
python3 -m src.maneuver_linker --no-motors --no-button
```

It **stops by design at the yaw-sign check**, because nothing turned. The console and `summary.txt` should say:

```
[MANEUVER] STOPPED: yaw sign unclear: ... the encoders didn't move either, so the motors or encoders aren't responding
```

It should also print a gyro bias line, write a run folder under `runs/maneuver_<timestamp>/`, and render `maneuver.avi`. If all of that happened, the robot is ready for the mat.

Raising the wheels doesn't help the spin check: the body has to turn for the gyro to see it, so a real trial needs the floor.

## 3. Running a trial on the mat

**Placement.** Put the robot at the end of the mat **furthest from the intersection**, centred in the lane and pointing along it. Clear the leg length ahead of it (see section 5), plus room to turn in place.

**First run: keep the leg short** and measure it:

```
python3 -m src.maneuver_linker --leg-counts 800
```

Then:

1. The console prints the output folder and the trial settings, and the display shows `rdy`.
2. **Press the start button.** The display counts down 5-4-3-2-1. Step back.
3. **Don't touch the robot for the first few seconds.** It sits still for 2 s (settling), then makes two short spins: left, then right.
4. It drives forward, stops for 1 s, turns left in place 180°, stops, drives back and stops. Every stop is a **short brake** (`MotorController.brake()`), not a coast, so it stops sharply; that's intended.
5. The video renders and plays if a screen is attached; close the window or wait for it to end. The summary prints at the end.

**Stopping it early:** Ctrl-C in the terminal. The motors stop first, and everything recorded so far is still saved and rendered. The run also stops itself if:
- a wheel is commanded but no encoder counts arrive for 0.5 s (stall);
- a frame arrives more than 0.5 s late;
- the turn passes 270°, or doesn't reach 180° within 6 s;
- the run passes 60 s (`--max-run-s`).

**Every time, measure and write down:**
- how far each leg actually travelled (tape measure, start to stop);
- the robot's actual heading after the turn: did it end up pointing back along the lane? Estimate the error in degrees if not;
- anything odd: a wheel slipping, a lurch, a pause.

### Stepped run: stopping to measure each step

```
python3 -m src.maneuver_linker --hold
```

The same trial, but the robot brakes and waits after each step until you press the **start button** (or Enter in the terminal). The console says what to measure at each stop. Take your time; waiting time doesn't count against `--max-run-s`, and nothing moves until you press. The camera and sensors keep recording during the wait.

| Hold | When | Measure |
| --- | --- | --- |
| `after_pulses` | after the two short spins | Is it pointing back along the start heading? Estimate the error in degrees |
| `after_leg_1` | after leg 1 stops | Distance travelled from the start line; how far it drifted sideways |
| `after_turn` | after the 180° turn settles | The angle turned, against the start heading (tape along the start heading makes this easy) |
| `after_leg_2` | after leg 2 stops | Distance travelled back; how far it stopped from the start mark |

Paper log to copy for each run (one line per hold):

```
Run folder: runs/maneuver_______________   Battery: ____   Surface: ____________
Settings changed (flags): _______________________________________

Hold           Measured by hand                        Robot said (summary.txt)
after_pulses   heading error ____ deg                  yaw sign ______
after_leg_1    distance ____ cm   drift ____ cm        counts L ____ R ____
after_turn     angle ____ deg                          final ____ deg   PASS / FAIL
after_leg_2    distance ____ cm   from start ____ cm   counts L ____ R ____
Notes:
```

`summary.txt` lists each hold with the encoder counts where it stopped, next to the leg and turn results, so the two columns can be filled from the same run.

## 4. Reading the results

Each run writes one folder, printed at the start as `output`:

```
runs/maneuver_<YYYYMMDD_HHMMSS>/
    summary.txt         the findings (below), then the Phase 2/3 timing and lane summaries
    report.json         the same findings, machine-readable
    maneuver.avi        the video: the Phase 3 view with a maneuver strip under it
    maneuver.csv        every frame: step, motor commands, corrections, encoder counts and
                        counts per second, yaw, heading, turn angle, lane status
    p3.csv              every frame's vision input and Phase 3 packet (as phase3_linker)
    config.json         the exact settings this run used
    frames/, records.pkl   what the video is made from
    error.txt           only if the program crashed: the traceback
```

`summary.txt` opens with the `[MANEUVER]` section:

| Line | What it means | What to look for |
| --- | --- | --- |
| `[MANEUVER] completed` / `STOPPED: ...` | Whether every step ran, or why it stopped | Any `STOPPED` reason |
| `gyro bias at rest` | The gyro's drift while still, measured this run, next to the configured value (−1.1) | Should be stable run to run. Noise sd above ~1 means the robot moved while settling |
| `lateral accel at rest` | Sideways acceleration while still | Not zero: the IMU mount's tilt. Note it; later work subtracts it |
| `yaw sign` | Which way a positive gyro reading turns, found by the two spins | **Report this value.** It settles a known disagreement in the code comments |
| `forward_1`, `forward_2` | Encoder counts per wheel, imbalance, time, whether it ended on counts or the time cap, gyro heading at the end, largest correction from each source | Heading at end near 0 means it drove straight. `time cap` means the leg was cut short (motors slow or `--leg-counts` too big) |
| `turn 180  PASS/FAIL` | Final gyro angle, time to reach it, overshoot | **PASS: within 180 ± 5° in under 6 s.** Compare with what you saw |
| `encoders during the turn` | Counts per wheel while turning, counts per degree | Report counts per degree; it calibrates future encoder-only turns |
| `lane found again` | How soon vision saw both lane lines after the turn | Should be well under a second; `never` is worth a note |
| `run` | Frames, time, FPS, dropped frames | FPS should be about 20 or more; any `recorder dropped` just means video gaps |

**In the video:** the top is the vision pipeline's view. The strip at the bottom shows the current step (amber settling and spins, green legs, cyan turn, red if it stopped), the motor commands, both corrections, wheel speeds and yaw. On a leg its bar fills to the leg length; in the turn it fills towards 180°, with the pass band in dark green.

## 5. Adjusting between runs

Every setting has a factory default in `MANEUVER` in `src/config.py`. For one run, override it on the command line; nothing is saved:

| Flag | Default | Effect |
| --- | --- | --- |
| `--leg-counts N` | 1500 | Leg length in encoder counts (mean of both wheels) |
| `--leg-max-s S` | 8 | Backstop: a leg stops after this long, even short of its counts |
| `--speed D` | 0.40 | Forward motor duty, 0–1 |
| `--turn-speed D` | 0.45 | Turning duty (it slows to 0.30 for the last 20°) |
| `--gyro-bias DPS` | −1.1 | Starting gyro bias; the settle step measures and replaces it |
| `--kp-counts K` | 0.0015 | How hard the encoders pull it straight |
| `--kp-heading K` | 0.01 | How hard the gyro pulls it straight |
| `--turn-tolerance DEG` | 5 | PASS band around 180° |
| `--turn-timeout S` | 6 | PASS time limit for the turn |
| `--max-run-s S` | 60 | Hard limit on the whole run |
| `--set FIELD=VALUE` | | Any other `ManeuverConfig` field, e.g. `--set turn_slow_band_deg=30`; repeatable |

**Sizing the leg to the mat:** after the first short run, counts per cm = `--leg-counts` ÷ the distance you measured. Pick a leg that leaves at least 30 cm before the intersection, and use that `--leg-counts` from then on. Report the counts per cm too.

Typical adjustments:
- **Turn overshoots and FAILs:** the robot brakes at 180°, so what's left is how far it turns while the brake bites. Check `maneuver.csv`: the `turn_settle` rows show how many degrees it gained after the stop. If that's still over ~4°, lower `--turn-speed`, or widen the slow-down band: `--set turn_slow_band_deg=30`. Report the numbers either way; they tell us whether the brake works as expected.
- **Drifts off straight:** raise `--kp-heading` (e.g. 0.02). If it wobbles side to side, lower it.
- **Leg ends by `time cap`:** raise `--leg-max-s`, or `--speed` if the motors are weak.

Once a value works reliably, tell whoever maintains `config.py` so it becomes the default.

Other flags: `--hold` (stop after each step to measure; see section 3), `--no-button` (starts after a 3 s console countdown; with `--hold`, press Enter to continue), `--no-display` (don't play the video), `--no-render` (skip the video), `--render RUN_DIR` (make the video later from a run folder), `--scale 2` (a bigger video), `--out DIR`.

## 6. Sending results back

For each run: the run folder name, your measured leg distances, the actual heading after the turn, and anything you noticed. Pack the folder:

```
tar -czf maneuver_results.tgz runs/maneuver_<YYYYMMDD_HHMMSS>
```

The `frames/` folder is the bulk (about 30 KB a frame). If the archive is too big, `summary.txt`, `report.json`, `maneuver.csv`, `p3.csv`, `config.json` and `maneuver.avi` are enough.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `hardware error: ... pigpio daemon not reachable` | `sudo pigpiod` |
| `hardware error` naming the camera | Something else has the camera; close it (`phase2_linker`, `rpicam-hello`, pytest) |
| `hardware error` naming `board` / `adafruit_mpu6050` / I²C | IMU libraries or I²C: `i2cdetect -y 1` must show 68 |
| `hardware error` naming `tm1637` | Display library missing; run with `--no-button` meanwhile |
| Nothing happens after starting | It's waiting for the start button (display shows `rdy`) |
| It stopped mid-trial and waits (`--hold`) | That's a hold: measure, then press the start button or Enter. A button still held from the start press is ignored until released |
| `STOPPED: yaw sign unclear ...` on the mat | The spins didn't turn the robot enough for the gyro. Check the battery; try `--set pulse_s=0.4` or `--set pulse_speed=0.5` |
| `STOPPED: stall` | A wheel was commanded but its encoder didn't count: check the encoder cables, or the wheel is blocked |
| `STOPPED: frame gap` | The Pi fell behind; close other programs. Report it with `summary.txt` |
| `STOPPED: no IMU readings while settling` | The IMU isn't answering; check wiring and `i2cdetect -y 1` |
| The first leg drives backward | Motor direction is reversed; stop and report it (see 2a) |
| Turn FAILs by a similar amount every run | Report the final angle and what you saw; that's the data we need, not a fault in the run |
| `error.txt` appears | The program crashed. Motors were stopped. Send the folder |
