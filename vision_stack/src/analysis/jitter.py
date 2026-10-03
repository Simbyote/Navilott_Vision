"""Frame-interval jitter: the tails and spikes a median hides.

Purpose:
    Linux isn't a real-time OS, so a steady median can sit on top of late
    frames that Navigation feels as steering hiccups. This reads per-frame
    intervals and reports the tail (p95, p99, max), how many frames missed
    the budget and in how long a streak, and whether the spikes are
    periodic, which points at a timer-driven cause (log flush, recorder
    write, a background service) rather than scene load.

Main package:
    analyze()      the numbers, plain dict
    jitter.png     interval per frame with the budget and spike markers,
                   beside a histogram of intervals

Input:
    Any CSV with interval_ms (measured loop), dt_ms (test_capture's
    frames.csv), dt_s (p3.csv) or timestamp_ms. See common.interval_ms.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import BUDGET_MS, Table, fmt, interval_ms, stats
from src.params import FPS

# nav.csv before p3.csv: in a navigation run folder its t is the loop interval (p3.csv's dt_s is clamped)
CSV_NAMES = ("frames.csv", "stage_timing.csv", "nav.csv", "p3.csv", "stages.csv")
# A spike sits this many robust standard deviations (1.4826 x MAD) above the
# median, and at least SPIKE_MIN_MS above it so a perfectly steady series
# doesn't call 0.1 ms of noise a spike
SPIKE_MADS = 5.0
SPIKE_MIN_MS = 2.0
WARMUP_FRAMES = 5

_CLI_HELP = """\
Frame-interval tails and spikes against the frame budget. Writes jitter.png
and jitter.json next to the input CSV.

Run from vision_stack/, venv active:
    python3 -m src.analysis.jitter                  newest run
    python3 -m src.analysis.jitter <folder or csv>  a specific run
"""


def spike_threshold(values: np.ndarray) -> float:
    """Median plus SPIKE_MADS robust deviations, and at least SPIKE_MIN_MS above the median."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med + max(SPIKE_MADS * 1.4826 * mad, SPIKE_MIN_MS)


def analyze(intervals, budget_ms: float = BUDGET_MS) -> dict:
    """
    Inputs:
        intervals: per-frame interval in ms; NaN entries are ignored
        budget_ms: frame budget

    Outputs:
        {
          "interval": stats(),
          "effective_fps": 1000 / mean interval,
          "over_budget": {"frames", "share", "longest_streak"},
          "spikes": {"threshold_ms", "count", "frames" (indices),
                     "spacing_frames": stats of gaps between spikes,
                     "periodic": True when the spacing is regular},
        }
    """
    iv = np.asarray(intervals, dtype=np.float64)
    valid = ~np.isnan(iv)
    s = stats(iv)
    if s["n"] == 0:
        raise ValueError("no intervals to analyze")

    over = valid & (iv > budget_ms)
    streak = max((n for flag, _, n in common.runs_of(over.tolist()) if flag), default=0)

    threshold = spike_threshold(iv[valid])
    spike_idx = np.flatnonzero(valid & (iv > threshold))
    spacing = np.diff(spike_idx)
    spacing_stats = stats(spacing)
    # Regular when spikes recur at a near-constant spacing: at least 3 gaps,
    # spread under a quarter of their median
    periodic = bool(spacing.size >= 3 and spacing_stats["p50"] > 1
                    and spacing_stats["std"] < 0.25 * spacing_stats["p50"])

    return {
        "budget_ms": budget_ms,
        "interval": s,
        "effective_fps": 1000.0 / s["mean"] if s["mean"] else None,
        "over_budget": {"frames": int(over.sum()), "share": float(over.sum() / s["n"]),
                        "longest_streak": int(streak)},
        "spikes": {"threshold_ms": float(threshold), "count": int(spike_idx.size),
                   "frames": spike_idx.tolist(), "spacing_frames": spacing_stats,
                   "periodic": periodic},
    }


def figure(intervals, result: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    iv = np.asarray(intervals, dtype=np.float64)
    budget = result["budget_ms"]
    fig, (ax, hx) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 1]})

    x = np.arange(iv.size)
    ax.plot(x, iv, color="#1f77b4", linewidth=0.8)
    ax.axhline(budget, color="black", linewidth=1.2, label=f"{budget:.0f} ms budget")
    ax.axhline(result["spikes"]["threshold_ms"], color="#d62728", linestyle=":",
               linewidth=1, label="spike threshold")
    idx = result["spikes"]["frames"]
    ax.scatter(idx, iv[idx], color="#d62728", s=12, zorder=3)
    ax.set_xlabel("frame")
    ax.set_ylabel("interval ms")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    v = iv[~np.isnan(iv)]
    # Identical intervals (a fixed-rate replay or a fake clock) have no range to split into 40 bins;
    # rounding leaves them differing by ~1e-14 ms, so anything under a microsecond is one bin
    hx.hist(v, bins=40 if np.ptp(v) > 1e-3 else 1, orientation="horizontal", color="#1f77b4")
    hx.axhline(budget, color="black", linewidth=1.2)
    hx.set_ylim(ax.get_ylim())
    hx.set_xlabel("frames")
    hx.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.jitter", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run folder or CSV (default: newest)")
    p.add_argument("--fps", type=float, default=FPS, help=f"target rate (default {FPS})")
    p.add_argument("--skip", type=int, default=WARMUP_FRAMES, help="drop the first N frames")
    p.add_argument("--out", help="output folder (default: next to the CSV)")
    args = p.parse_args(argv)

    try:
        path = common.find_csv(args.run, CSV_NAMES)
        iv = interval_ms(Table(path, args.skip))
        if iv is None:
            raise ValueError(f"{path.name}: no interval or timestamp column")
        res = analyze(iv, 1000.0 / args.fps)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else path.parent
    common.write_json(out / "jitter.json", res)
    fig = figure(iv, res, f"Frame interval: {path.parent.name}", out / "jitter.png")

    s, ob, sp = res["interval"], res["over_budget"], res["spikes"]
    print(f"{path}  ({s['n']} intervals, budget {res['budget_ms']:.1f} ms)")
    print(f"  interval ms   median {fmt(s['p50'])}  p95 {fmt(s['p95'])}  "
          f"p99 {fmt(s['p99'])}  max {fmt(s['max'])}")
    print(f"  effective     {fmt(res['effective_fps'], 1)} FPS")
    print(f"  over budget   {ob['frames']} frames ({100 * ob['share']:.1f}%), "
          f"longest streak {ob['longest_streak']}")
    spacing = sp["spacing_frames"]["p50"]
    print(f"  spikes        {sp['count']} over {fmt(sp['threshold_ms'], 1)} ms"
          + (f", every ~{spacing:.0f} frames (periodic)" if sp["periodic"] else ""))
    print(f"wrote {out / 'jitter.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
