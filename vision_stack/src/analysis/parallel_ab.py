"""Serial against parallel perception over whole runs, alternated.

Purpose:
    parallel_compare times Phase 2 alone on frames held in memory; a run
    also shares the four cores and the GIL with the sensor hub, the
    recorder, Phase 3 and navigation. So this runs whole navigation runs,
    serial and parallel in turn (S, P, S, P, ...) so heat and the battery
    affect both alike, and compares them (docs/guides/analysis.md).

Main package:
    ab.json in --out: per run and mode, the frame rate, time between frames,
    Phase 2, the share over budget and frame-to-motor latency; the verdict,
    including whether every parallel run beat every serial one.

Flow:
    1. --runs navigation runs, each --seconds, alternating modes, each with
       a --label; dry by default, --motors waits for Enter before each.
    2. Each run's nav.csv and report.json read; modes compared.
    (--compare reads runs made before instead.)

Usage (from vision_stack/, on the Pi, pigpiod running):
    python3 -m src.analysis.parallel_ab                       # 4 runs of 60 s, dry
    python3 -m src.analysis.parallel_ab --runs 6 --seconds 120
    python3 -m src.analysis.parallel_ab --compare runs/ab_<time>/*    # compare runs made before
    make ab-parallel RUNS=4 SECONDS=60
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from src.params import RUNS_DIR

RUNS = 4
SECONDS = 60.0
BUDGET_MS = 50.0
SERIAL, PARALLEL = "serial", "parallel"
METRICS = ("fps", "interval_p50", "interval_p95", "phase2_p50", "phase2_p95", "over_budget_pct",
           "latency_p50", "latency_p95")


def _num(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def run_stats(folder: Path) -> dict:
    """
    One run's numbers from its nav.csv (frame 0, the start-up, left out)
    and report.json (its mode and label).

    Raises:
        FileNotFoundError: No nav.csv in the folder.
    """
    folder = Path(folder)
    with open(folder / "nav.csv") as f:
        rows = list(csv.DictReader(f))[1:]
    report = {}
    if (folder / "report.json").exists():
        report = json.loads((folder / "report.json").read_text())
    t = [x for x in (_num(r.get("t")) for r in rows) if x is not None]
    interval = [(b - a) * 1000.0 for a, b in zip(t, t[1:])]
    p2 = [x for x in (_num(r.get("phase2_ms")) for r in rows) if x is not None]
    work = [a + b for a, b in ((_num(r.get("phase2_ms")), _num(r.get("phase3_ms"))) for r in rows)
            if a is not None and b is not None]
    latency = [x for x in (_num(r.get("latency_ms")) for r in rows) if x is not None]
    span = t[-1] - t[0] if len(t) > 1 else 0.0
    return {"run": folder.name, "mode": PARALLEL if report.get("parallel_perception") else SERIAL,
            "label": report.get("label", ""), "frames": len(rows),
            "fps": round((len(t) - 1) / span, 2) if span > 0 else None,
            "interval_p50": _pct(interval, 50), "interval_p95": _pct(interval, 95),
            "phase2_p50": _pct(p2, 50), "phase2_p95": _pct(p2, 95),
            "over_budget_pct": round(100.0 * sum(w > BUDGET_MS for w in work) / len(work), 1) if work else None,
            "latency_p50": _pct(latency, 50), "latency_p95": _pct(latency, 95)}


def compare(stats: list[dict]) -> dict:
    """
    The modes side by side: each metric's mean over the mode's runs, the
    change parallel minus serial, and whether the time between frames
    separates them (every parallel run's p50 under every serial run's).
    """
    by = {m: [s for s in stats if s["mode"] == m] for m in (SERIAL, PARALLEL)}
    means = {m: {k: (round(statistics.fmean(v), 2) if (v := [s[k] for s in runs if s[k] is not None]) else None)
                 for k in METRICS} for m, runs in by.items()}
    change = {k: (None if None in (means[SERIAL][k], means[PARALLEL][k])
                  else round(means[PARALLEL][k] - means[SERIAL][k], 2)) for k in METRICS}
    ser = [s["interval_p50"] for s in by[SERIAL] if s["interval_p50"] is not None]
    par = [s["interval_p50"] for s in by[PARALLEL] if s["interval_p50"] is not None]
    separated = bool(ser and par) and max(par) < min(ser)
    if not (ser and par):
        verdict = "not compared: both modes need at least one run"
    elif separated:
        verdict = (f"parallel faster in every run: {change['interval_p50']:+.1f} ms between frames, "
                   f"{change['fps']:+.2f} FPS")
    elif min(par) > max(ser):
        verdict = f"parallel slower in every run: {change['interval_p50']:+.1f} ms between frames"
    else:
        verdict = (f"no clear difference: the runs overlap ({change['interval_p50']:+.1f} ms between frames "
                   f"on the means); run more, or longer")
    return {"runs": stats, "means": means, "change": change, "separated": separated, "verdict": verdict}


def report_lines(result: dict) -> list[str]:
    """The printed comparison."""
    head = f"  {'run':<28} {'mode':<9}" + "".join(f"{k:>16}" for k in METRICS)
    out = [head]
    for s in result["runs"]:
        out.append(f"  {s['run']:<28} {s['mode']:<9}"
                   + "".join(f"{'--' if s[k] is None else s[k]:>16}" for k in METRICS))
    for m in (SERIAL, PARALLEL):
        out.append(f"  {'mean':<28} {m:<9}"
                   + "".join(f"{'--' if result['means'][m][k] is None else result['means'][m][k]:>16}"
                             for k in METRICS))
    out.append(f"  {'parallel - serial':<38}"
               + "".join(f"{'--':>16}" if result["change"][k] is None else f"{result['change'][k]:>+16}"
                         for k in METRICS))
    out.append(f"\nVerdict: {result['verdict']}")
    return out


def plan(runs: int) -> list[str]:
    """The modes in order, alternating from serial: S, P, S, P, ..."""
    return [SERIAL if i % 2 == 0 else PARALLEL for i in range(runs)]


def launch_args(mode: str, k: int, out: Path, seconds: float, motors: bool, extra: list[str]) -> list[str]:
    """navigation_linker's command line for run k (1-based) in this mode."""
    return (["--camera", "--no-button", "--no-render", "--no-display", "--max-run-s", f"{seconds:g}",
             f"--{mode}", "--label", f"A/B {mode} {k}", "--out", str(out)]
            + ([] if motors else ["--no-motors"]) + extra)


def main(argv: list[str] | None = None, launch=None, ask=input, say=print) -> int:
    ap = argparse.ArgumentParser(prog="parallel_ab", description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=RUNS, help=f"runs in all, alternating serial and parallel ({RUNS})")
    ap.add_argument("--seconds", type=float, default=SECONDS, help=f"each run's length ({SECONDS:g})")
    ap.add_argument("--motors", action="store_true", help="drive each run (Enter before each); default dry")
    ap.add_argument("--out", default=None, metavar="DIR", help="where the runs and ab.json go")
    ap.add_argument("--compare", nargs="+", default=None, metavar="RUN_DIR",
                    help="compare these finished runs instead of making new ones")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="after --, more navigation_linker arguments for every run (e.g. -- --no-record)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    extra = [a for a in args.extra if a != "--"]

    if args.compare:
        folders = [Path(p) for p in args.compare]
        out = Path(args.out) if args.out else Path(".")
    else:
        if args.runs < 2:
            say("ERROR: --runs needs at least 2 (one of each mode)")
            return 2
        if launch is None:
            from src.navigation_linker import cli as launch
        out = Path(args.out) if args.out else RUNS_DIR / ("ab_" + time.strftime("%Y%m%d_%H%M%S"))
        folders = []
        for k, mode in enumerate(plan(args.runs), 1):
            folder = out / f"{k:02d}_{mode}"
            say(f"\n=== run {k} of {args.runs}: {mode}, {args.seconds:g} s ===")
            if args.motors:
                ask("Robot at the start, in its lane? Enter drives it: ")
            code = launch(launch_args(mode, k, folder, args.seconds, args.motors, extra))
            if code != 0:
                say(f"ERROR: run {k} ended with code {code}; stopping the A/B")
                return code
            folders.append(folder)

    try:
        stats = [run_stats(f) for f in folders]
    except FileNotFoundError as exc:
        say(f"ERROR: {exc}")
        return 2
    result = compare(stats)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab.json").write_text(json.dumps(result, indent=2))
    say("\n".join(report_lines(result)))
    say(f"\nab.json written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
