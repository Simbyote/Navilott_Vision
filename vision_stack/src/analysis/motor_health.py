"""Motor health: how fast each wheel turns for the duty it's given, across a run.

Purpose:
    A failing motor doesn't stop at once: the wheel-speed PI raises its duty
    to hold the target, so the robot drives on while the motor weakens. Its
    counts/s per unit of duty doesn't hide it. On 2026-10-28 the right N20
    read ~3300 for 7.5 min, fell 25% in one turn at 455 s, and stopped at
    587 s. This finds that drop and draws it, for nav-report and fig8-report.

Main package:
    analyze(table) -> per wheel: baseline, windows, first weak window,
    seconds driven but not turning; findings; draw(ax, res).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table, runs_of
from src.navigation.wheel_stall import MIN_DUTY, MOVING_SHARE

CSV_NAMES = ("nav.csv",)
NEEDS = ("t", "cmd_left", "cmd_right", "left_cps", "right_cps")
WINDOW_S = 5.0
# The baseline is the run's first quarter (at least BASELINE_MIN_WINDOWS windows): the motor as it started
BASELINE_SHARE = 0.25
BASELINE_MIN_WINDOWS = 3
MIN_FRAMES = 10                 # driven frames for a window to count
# A window this far under the baseline is weak. Healthy windows on 2026-10-28 dipped 11% at most
WEAK_SHARE = 0.15
# The two wheels' baselines further apart than this: one motor started weaker
MISMATCH_SHARE = 0.10
# Driven and not turning counts in stretches this long: a start from rest reads so for ~50 ms
STOPPED_MIN_S = 0.3
COLORS = {"left": "#1f77b4", "right": "#d62728"}

_CLI_HELP = """\
How fast each wheel turned for the duty it was given (counts/s per unit duty),
in 5 s windows across a navigation run, against how it started. A drop is a
motor or gearbox failing; a wheel driven but not turning is a stall.

    make motor-health                      the newest run
    make motor-health RUN=runs/nav_<time>  or a routine attempt with a nav.csv

Writes motor_health.png and motor_health.json next to nav.csv. nav-report and
fig8-report draw the same panel.
"""


def wheel(t: np.ndarray, duty: np.ndarray, cps: np.ndarray, brake: np.ndarray) -> dict:
    """One wheel. Each frame's speed answers the previous frame's command, as the encoders' window does."""
    prev = np.r_[np.nan, duty[:-1]]
    braked = np.r_[True, brake[:-1] == 1]
    driven = (np.abs(prev) >= MIN_DUTY) & ~braked & np.isfinite(cps) & np.isfinite(t)
    forward = np.where(prev > 0, cps, -cps)             # turning the wrong way reads negative
    resp = np.full(len(t), np.nan)
    resp[driven] = forward[driven] / np.abs(prev[driven])
    edges = np.arange(np.nanmin(t), np.nanmax(t) + WINDOW_S, WINDOW_S) if np.isfinite(t).any() else np.array([])
    windows = []
    for a in edges[:-1]:
        sel = driven & (t >= a) & (t < a + WINDOW_S)
        if sel.sum() >= MIN_FRAMES:
            windows.append({"t": round(float(a + WINDOW_S / 2), 2), "frames": int(sel.sum()),
                            "response": round(float(np.median(resp[sel])), 1)})
    out = {"windows": windows, "baseline": None, "worst": None, "weak_from_s": None, "weak_windows": 0,
           "stopped_s": 0.0, "stopped_from_s": None}
    if not windows:
        return out
    n_base = max(BASELINE_MIN_WINDOWS, int(np.ceil(BASELINE_SHARE * len(windows))))
    base = float(np.median([w["response"] for w in windows[:n_base]]))
    worst = min(windows, key=lambda w: w["response"])
    weak = [w for w in windows if w["response"] < (1.0 - WEAK_SHARE) * base]
    still = driven & (forward < MOVING_SHARE * np.abs(prev) * base)
    dt = np.r_[0.0, np.clip(np.diff(t), 0.0, 0.5)]
    stopped = np.zeros(len(t), bool)
    for v, start, length in runs_of(still.tolist()):
        if v and t[start + length - 1] - t[start] + dt[start] >= STOPPED_MIN_S:
            stopped[start:start + length] = True
    out.update({"baseline": round(base, 1), "worst": {"t": worst["t"], "response": worst["response"],
                                                     "share": round(worst["response"] / base, 2) if base else None},
                "weak_from_s": weak[0]["t"] if weak else None, "weak_windows": len(weak),
                "stopped_s": round(float(dt[stopped].sum()), 2),
                "stopped_from_s": round(float(t[stopped][0]), 2) if stopped.any() else None})
    return out


def analyze(table: Table) -> dict:
    """Both wheels of a nav.csv; ValueError without its wheel columns."""
    missing = [c for c in NEEDS if not table.has(c)]
    if missing:
        raise ValueError(f"{table.path.name}: no {', '.join(missing)} (not a nav.csv)")
    t = table.numeric("t")
    brake = table.numeric("brake") if table.has("brake") else np.zeros(len(t))
    res = {side: wheel(t, table.numeric(f"cmd_{side}"), table.numeric(f"{side}_cps"), brake)
           for side in ("left", "right")}
    rules = table.text("rule") if table.has("rule") else []
    trips = [i for i, r in enumerate(rules) if r == "wheel_stall"]
    res["stall_guard_s"] = round(float(t[trips[0]]), 2) if trips else None
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    out = []
    for side in ("left", "right"):
        w = res[side]
        if w["baseline"] is None:
            continue
        if w["weak_from_s"] is not None:
            out.append(f"the {side} wheel weakened from {w['weak_from_s']:.0f} s: {w['weak_windows']} windows "
                       f"over {100 * WEAK_SHARE:.0f}% under how it started ({w['baseline']:.0f} counts/s per duty), "
                       f"worst {100 * w['worst']['share']:.0f}% at {w['worst']['t']:.0f} s. The motor or its "
                       "gearbox is failing: check it before the next run")
        if w["stopped_s"] >= 0.25:
            out.append(f"the {side} wheel was driven but not turning for {w['stopped_s']:.1f} s, from "
                       f"{w['stopped_from_s']:.0f} s: a stall, a jam or a dead motor")
    a, b = res["left"]["baseline"], res["right"]["baseline"]
    if a and b and abs(a - b) > MISMATCH_SHARE * max(a, b):
        weaker = "left" if a < b else "right"
        out.append(f"the {weaker} motor started {100 * abs(a - b) / max(a, b):.0f}% weaker than the other "
                   f"({a:.0f} vs {b:.0f} counts/s per duty)")
    if res.get("stall_guard_s") is not None:
        out.append(f"the stall guard ended the run at {res['stall_guard_s']:.0f} s")
    return out


def report_lines(res: dict) -> list[str]:
    lines = []
    for side in ("left", "right"):
        w = res[side]
        if w["baseline"] is None:
            lines.append(f"{side:<6} wheel  no driven windows")
            continue
        lines.append(f"{side:<6} wheel  {w['baseline']:.0f} counts/s per duty at the start; worst "
                     f"{w['worst']['response']:.0f} ({100 * w['worst']['share']:.0f}%) at {w['worst']['t']:.0f} s; "
                     f"driven but not turning {w['stopped_s']:.1f} s")
    return lines


def drawable(res: dict | None) -> bool:
    """Whether either wheel has a window to draw: a run that logged its duties and drove."""
    return bool(res) and any(res[side]["windows"] for side in ("left", "right"))


def draw(ax, res: dict) -> None:
    """Each wheel's response per window, its baseline dotted, the weak band shaded below it."""
    for side in ("left", "right"):
        w = res[side]
        if not w["windows"]:
            continue
        ax.plot([x["t"] for x in w["windows"]], [x["response"] for x in w["windows"]], color=COLORS[side],
                lw=1.1, marker=".", ms=3, label=f"{side} wheel")
        ax.axhline(w["baseline"], color=COLORS[side], ls=":", lw=0.8)
        ax.axhspan(0, (1.0 - WEAK_SHARE) * w["baseline"], color=COLORS[side], alpha=0.04, lw=0)
        if w["weak_from_s"] is not None:
            ax.axvline(w["weak_from_s"], color=COLORS[side], ls="--", lw=0.8)
        if w["stopped_from_s"] is not None:
            ax.axvline(w["stopped_from_s"], color="k", ls=":", lw=0.9)
    if res.get("stall_guard_s") is not None:
        ax.axvline(res["stall_guard_s"], color="k", lw=1.2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"counts/s per duty\n({WINDOW_S:.0f} s median)", fontsize=8)
    if drawable(res):
        ax.legend(fontsize=7, frameon=False, loc="lower left")


def figure(res: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(12, 3.6))
    draw(ax, res)
    ax.set_xlabel(f"run time, s (dotted: each wheel's start; dashed: first window {100 * WEAK_SHARE:.0f}% under it; "
                  "black dotted: driven, not turning; black: the stall guard)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(alpha=0.25)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.motor_health", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run folder or nav.csv (default: newest nav.csv)")
    p.add_argument("--out", help="output folder (default: next to nav.csv)")
    args = p.parse_args(argv)
    try:
        path = common.find_csv(args.run, CSV_NAMES)
        res = analyze(Table(path))
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else path.parent
    common.write_json(out / "motor_health.json", res)
    fig = figure(res, f"Motor health: {path.parent.name}", out / "motor_health.png")
    print(f"{path}\n" + "\n".join(report_lines(res)))
    print("findings\n" + "\n".join(f"  - {f}" for f in res["findings"] or ["nothing stood out"]))
    print(f"wrote {out / 'motor_health.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
