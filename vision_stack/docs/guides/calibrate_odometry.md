# Odometry Calibration (cm per Encoder Count)

> How many floor cm one encoder count is, from a taped distance.

Odometry turns the wheel encoders' counts into distance driven (`distance_cm` in the packet), so the rules can measure in cm instead of time. It needs one number, **cm per count**. It depends on the wheels (diameter, how much the tyres squash), the gearbox and how the encoders are decoded, so it's measured rather than computed: move the robot a known distance, count, divide.

It writes `calibration/odometry.json`. `MEASURED_ESTIMATION` (`config.py`) reads `cm_per_count` from it when the pipeline starts. Without the file it uses the nominal scale in `params.py`: a 3.3 cm wheel (`WHEEL_DIAMETER_CM`) over 1400 counts per wheel turn (`ENCODER_COUNTS_PER_REV`: a 7-pulse encoder behind the N20s' 200:1 gearbox, sold as 12 V, 140 RPM no load), about 0.0074 cm per count. Measured beats nominal: tyres squash, and the encoder's 7 pulses per turn are assumed. Nothing drives by distance until its `*_CM` setting is set either way.

**Code:** `src/scripts/calibrate_odometry.py`, `src/estimation/odometry.py` · **Tests:** `test_calibrate_odometry.py`, `test_odometry.py`

---

## How it works

The encoders count every edge of both quadrature channels (`drive.EncoderReader`), and the sensor hub reads the running totals 100 times a second. Odometry takes each frame's totals, subtracts the first frame's, and multiplies by cm per count. Each wheel gets its own distance; the robot's is their mean (the middle of the axle), so a spin in place adds about nothing.

It uses the running totals, never speed × frame time, so a slow frame loses no distance.

---

## Requirements

- `sudo pigpiod` running (the encoders are read through pigpio).
- A straight stretch of floor at least 1.2 m long, the same surface the robot drives on.
- A tape measure and two strips of tape.

---

## 1. Push trials (the default)

The motors stay off and you roll the robot by hand. The distance is exact and nothing slips; the N20 gearboxes turn stiffly, so push slowly and keep it straight.

1. Put two tape marks 100 cm apart (`--cm` for another distance).
2. `make calib-odometry` (or `python3 -m src.scripts.calibrate_odometry`).
3. For each trial (3 by default, `--trials`): set the axle on the first mark, Enter; roll it straight to the second mark, Enter.

Each trial prints both wheels' counts and counts per cm. Use the same point on the robot at both marks (the axle, or the front edge).

No tape measure? Use the wheel: mark it, roll the robot exactly 10 turns, and give that distance, 10 × π × 3.3 = 103.7 cm: `make calib-odometry ARGS="--cm 103.7"`.

---

## 2. Drive trials (`--drive S`)

The robot drives itself straight at `--duty` (0.40, lane keeping's base duty) for S seconds, brakes, and asks how far it went. This includes the wheel slip the robot really has at driving speed.

```
make calib-odometry ARGS="--drive 3"
```

Mark where the axle starts, press Enter, and measure from the mark to the axle once it stops. The motors only drive while the trial runs; Ctrl-C or `q` stops.

Push trials and drive trials should agree within a few %. If drive trials read noticeably more cm per count than push trials, the wheels slip on that floor.

---

## Reading the result

```
cm per count   0.007570  (132.100 counts/cm)
each wheel     left 131.600, right 132.600 counts/cm
trials         3, spread 0.8% between the largest and smallest
vs nominal     +2.2% (3.3 cm wheel, 1400 counts/turn): a 3.373 cm wheel, or 1369.5 counts/turn
```

- **cm per count** is what odometry uses.
- **Each wheel** should be within a few %. A trial whose wheels differ by more than 3% is flagged: the push wasn't straight, or a wheel slipped. Redo it.
- **vs nominal** compares it with the wheel and gearbox's scale, as the wheel diameter that would explain it (at 1400 counts per turn) and the counts per turn (at 3.3 cm). A few % is tyre squash and slip. 2× or more (flagged) is the gear ratio or the decoding, not the tyre: set `ENCODER_COUNTS_PER_REV` from the counts per turn.
- **Spread** is how repeatable the trials were. Under ~2% is good; more means the pushes weren't consistent. Do more trials or a longer distance.

`--dry-run` prints without writing the file.

---

## When to redo it

- New wheels or tyres, or a different gearbox.
- A change to how the encoders are decoded (`drive.EncoderReader`).
- A different floor, if drive trials there disagree with the old result.

Battery voltage doesn't change it: a count is a fixed piece of a wheel turn, however fast the wheel turns.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `wheels counted 0, 0` | pigpiod isn't running (`sudo pigpiod`), or an encoder isn't connected |
| A wheel counts negative | That encoder's channels are swapped, or the robot was pushed backwards. Check with `maneuver_linker`'s spin checks |
| Wheels differ by more than 3% on every trial | One wheel slips or drags, or the robot drifts sideways when pushed. Push along a straight edge |
| `distance_cm` blank in nav.csv | No `calibration/odometry.json`, or the run had no encoders (`--encoders`, or a replay) |
