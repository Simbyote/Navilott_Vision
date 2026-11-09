"""Odometry calibration: floor cm per encoder count, from measured runs.

Purpose:
    One number turns counts into cm, and it depends on the wheels, the
    gearbox and the decoding, so it's measured: move a measured distance,
    count, divide. Pushed by hand (no slip) or --drive (the slip it really
    has); both wheels should agree within a few percent.

Main package:
    calibration/odometry.json: cm_per_count, each wheel's, and the trials' spread.
"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

from src.params import ENCODER_COUNTS_PER_REV, NOMINAL_CM_PER_COUNT, ODOMETRY_PATH, WHEEL_DIAMETER_CM

DEFAULT_CM = 100.0
DEFAULT_TRIALS = 3
DEFAULT_DUTY = 0.40           # lane keeping's BASE_SPEED
# Under the motor watchdog's 0.5 s
DRIVE_RESEND_S = 0.1
# Past this the push wasn't straight or a wheel slipped; reported, still counted
WHEEL_MISMATCH_PCT = 3.0

_CLI_HELP = """\
examples:
  python3 -m src.scripts.calibrate_odometry                     # push 100 cm by hand, 3 trials
  python3 -m src.scripts.calibrate_odometry --cm 150 --trials 5
  python3 -m src.scripts.calibrate_odometry --drive 3           # drive 3 s, then tape-measure
  make calib-odometry ARGS="--drive 3"
"""


def trial(cm: float, before: tuple[int, int], after: tuple[int, int]) -> dict:
    """
    One measured move. ValueError for a distance that isn't positive, or wheels that counted backwards or not at all.
    """
    if cm <= 0:
        raise ValueError(f"distance must be positive, got {cm}")
    left, right = after[0] - before[0], after[1] - before[1]
    if left <= 0 or right <= 0:
        raise ValueError(f"wheels counted {left}, {right}: both must count forward "
                         "(is pigpiod running, did the robot move forward?)")
    mean = (left + right) / 2.0
    return {"cm": cm, "left": left, "right": right, "counts_per_cm": round(mean / cm, 4),
            "mismatch_pct": round(100.0 * abs(left - right) / mean, 2)}


def summarize(trials: list[dict]) -> dict:
    """
    cm_per_count, each wheel's counts_per_cm, and spread_pct: how repeatable the trials were. ValueError with none.
    """
    if not trials:
        raise ValueError("no trials")
    cm = sum(t["cm"] for t in trials)
    left = sum(t["left"] for t in trials) / cm
    right = sum(t["right"] for t in trials) / cm
    mean = (left + right) / 2.0
    per = [t["counts_per_cm"] for t in trials]
    return {"cm_per_count": round(1.0 / mean, 6), "counts_per_cm": round(mean, 4),
            "left_counts_per_cm": round(left, 4), "right_counts_per_cm": round(right, 4),
            "spread_pct": round(100.0 * (max(per) - min(per)) / mean, 2), "trials": trials}


def run(counts, ask=input, say=print, cm: float = DEFAULT_CM, trials: int = DEFAULT_TRIALS,
        motor=None, drive_s: float | None = None, duty: float = DEFAULT_DUTY, sleep=time.sleep) -> list[dict]:
    """
    One trial() per move kept. A failed trial is said and redone; q stops. drive_s drives instead of pushing.
    """
    done = []
    while len(done) < trials:
        n = len(done) + 1
        if drive_s is None:
            if ask(f"\ntrial {n}/{trials}: axle (or front edge) on the 0 mark, motors off. Enter (q stops): ").strip() == "q":
                break
            before = counts()
            if ask(f"  roll it straight to the {cm:g} cm mark by hand. Enter: ").strip() == "q":
                break
            after, moved = counts(), cm
        else:
            if ask(f"\ntrial {n}/{trials}: mark where the axle is; {drive_s:g} s straight at duty {duty:g}. "
                   "Enter drives (q stops): ").strip() == "q":
                break
            before = counts()
            try:
                left = drive_s
                while left > 0:
                    motor.drive(duty, duty)
                    sleep(min(DRIVE_RESEND_S, left))
                    left -= DRIVE_RESEND_S
            finally:
                motor.brake()
            sleep(0.5)                          # coast to a stop before counting
            after = counts()
            typed = ask("  how far did it go, cm (axle to axle)? ").strip()
            if typed == "q":
                break
            try:
                moved = float(typed)
            except ValueError:
                say(f"  not a number: {typed!r}; redo this trial")
                continue
        try:
            t = trial(moved, before, after)
        except ValueError as e:
            say(f"  {e}; redo this trial")
            continue
        flag = f"  (wheels {t['mismatch_pct']:g}% apart: not straight, or a wheel slipped)" \
            if t["mismatch_pct"] > WHEEL_MISMATCH_PCT else ""
        say(f"  left {t['left']}, right {t['right']} counts over {moved:g} cm: {t['counts_per_cm']:g} counts/cm{flag}")
        done.append(t)
    return done


def against_nominal(s: dict, nominal: float = NOMINAL_CM_PER_COUNT) -> dict:
    """The measured scale against the nominal, as a wheel diameter and as counts per turn."""
    cpc = s["cm_per_count"]
    return {"vs_nominal_pct": round(100.0 * (cpc - nominal) / nominal, 1),
            "effective_diameter_cm": round(cpc * ENCODER_COUNTS_PER_REV / math.pi, 3),
            "counts_per_rev": round(math.pi * WHEEL_DIAMETER_CM / cpc, 1)}


def report(s: dict) -> str:
    n = against_nominal(s)
    return "\n".join([
        f"cm per count   {s['cm_per_count']:.6f}  ({s['counts_per_cm']:.3f} counts/cm)",
        f"each wheel     left {s['left_counts_per_cm']:.3f}, right {s['right_counts_per_cm']:.3f} counts/cm",
        f"trials         {len(s['trials'])}, spread {s['spread_pct']:g}% between the largest and smallest",
        f"vs nominal     {n['vs_nominal_pct']:+g}% ({WHEEL_DIAMETER_CM:g} cm wheel, {ENCODER_COUNTS_PER_REV} counts/turn): "
        f"a {n['effective_diameter_cm']:g} cm wheel, or {n['counts_per_rev']:g} counts/turn"
        + ("  (off by 2x or more: the gear ratio or the decoding, not the tyre)"
           if abs(n["vs_nominal_pct"]) >= 50 else ""),
    ])


def main(argv: list[str] | None = None, hardware=None, ask=input, say=print) -> int:
    """hardware: () -> (counts, motor or None, close), for tests; None opens the real encoders."""
    p = argparse.ArgumentParser(prog="calibrate_odometry", description="Floor cm per encoder count, from measured runs.",
                                epilog=_CLI_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cm", type=float, default=DEFAULT_CM, help="distance pushed per trial (default %(default)s)")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--drive", type=float, default=None, metavar="S", help="drive S seconds instead of pushing")
    p.add_argument("--duty", type=float, default=DEFAULT_DUTY, help="duty while driving (default %(default)s)")
    p.add_argument("--out", default=str(ODOMETRY_PATH))
    p.add_argument("--dry-run", action="store_true", help="print, don't write")
    a = p.parse_args(argv)

    counts, motor, close = (hardware or (lambda: _open(a.drive is not None)))()
    try:
        trials = run(counts, ask, say, cm=a.cm, trials=a.trials, motor=motor, drive_s=a.drive, duty=a.duty)
    finally:
        close()
    if not trials:
        say("no trials kept; nothing written")
        return 1
    s = summarize(trials)
    say("\n" + report(s))
    if a.dry_run:
        return 0
    out = {**s, "mode": "drive" if a.drive is not None else "push",
           **({"drive_s": a.drive, "duty": a.duty} if a.drive is not None else {}),
           "measured": datetime.now().isoformat(timespec="seconds")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    say(f"wrote {a.out}: the pipeline reads it at its next start")
    return 0


def _open(with_motors: bool):
    import pigpio
    from src.peripherals.drive import EncoderReader, MotorController
    pi = pigpio.pi()
    enc = EncoderReader(pi)
    motor = MotorController(pi) if with_motors else None

    def close():
        if motor is not None:
            motor.stop()
        enc.cancel()
        pi.stop()
    return enc.counts, motor, close


if __name__ == "__main__":
    raise SystemExit(main())
