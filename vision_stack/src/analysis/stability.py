"""Still-scene stability: offset noise and lane-mode flicker with nothing moving.

Purpose:
    Park the robot centered in the lane and record. Anything that changes
    is the pipeline, not the world: the offset's spread is the measurement
    noise floor Navigation steers through, and every lane-mode change is a
    flicker (two_boundary -> left_only -> two_boundary) that would jolt a
    controller. Run it again after tuning to see whether the floor dropped.

Main package:
    analyze()        the numbers, plain dict
    stability.png    offset per frame over a lane-mode strip, beside the
                     offset histogram

Input (first match wins):
    p3.csv                   lane_offset_cm on frames with lane_status
                             "vision" (a held value repeated for frames would
                             understate the spread); p2_mode for the modes
    lane_offset_timing.csv,
    stages.csv               offset and mode, normalized units
    --normalized uses the normalized offset even when cm are available.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table, fmt, runs_of, stats

CSV_NAMES = ("p3.csv", "lane_offset_timing.csv", "stages.csv")
MEASURED_MODES = frozenset({"two_boundary", "left_only", "right_only"})
FLICKER_FRAMES = 2          # a mode run this short between two others is a flicker
WARMUP_FRAMES = 5
MODE_COLORS = {"two_boundary": "#2ca02c", "left_only": "#1f77b4", "right_only": "#9467bd",
               "single_uncalibrated": "#ff7f0e", "none": "#d62728"}

_CLI_HELP = """\
Offset noise and lane-mode flicker on a still scene. Record with the robot
parked, centered, and nothing moving. Writes stability.png and
stability.json next to the input CSV.

Run from vision_stack/, venv active:
    python3 -m src.analysis.stability <folder or csv>
"""


def load(table: Table, normalized: bool = False) -> tuple:
    """
    (offset array, measured mask, modes list, unit) from whichever columns
    the table has; see the module docstring for the order.
    """
    modes = table.text("p2_mode") if table.has("p2_mode") else (
        table.text("mode") if table.has("mode") else None)
    if modes is None:
        raise ValueError(f"{table.path.name}: no mode or p2_mode column")

    if not normalized and table.has_values("lane_offset_cm"):
        offset = table.numeric("lane_offset_cm")
        measured = (np.array([s == "vision" for s in table.text("lane_status")])
                    if table.has("lane_status") else
                    np.array([m in MEASURED_MODES for m in modes]))
        unit = "cm"
    else:
        col = table.first_with_values("p2_offset", "offset")
        if col is None:
            raise ValueError(f"{table.path.name}: no offset column with values")
        offset = table.numeric(col)
        measured = np.array([m in MEASURED_MODES for m in modes])
        unit = "normalized"
    return offset, measured & ~np.isnan(offset), modes, unit


def analyze(offset, measured, modes, unit: str = "normalized") -> dict:
    """
    Inputs:
        offset: per-frame offset; only frames where measured is True count
        measured: per-frame bool
        modes: per-frame lane mode strings
        unit: "cm" or "normalized", carried into the result

    Outputs:
        {
          "frames", "unit",
          "measured": {"frames", "share"},
          "offset": stats() over measured frames, plus "p5_p95_span",
          "modes": {mode: {"frames", "share"}},
          "transitions": {"count", "per_100_frames"},
          "flickers": runs of FLICKER_FRAMES or fewer between two other runs,
          "longest_unmeasured_run": frames,
        }
    """
    offset = np.asarray(offset, dtype=np.float64)
    measured = np.asarray(measured, dtype=bool)
    n = len(modes)
    if n == 0:
        raise ValueError("no frames to analyze")

    values = offset[measured]
    off = stats(values)
    off["p5_p95_span"] = (float(np.percentile(values, 95) - np.percentile(values, 5))
                          if values.size else None)

    counts = Counter(modes)
    runs = runs_of(modes)
    flickers = sum(1 for i, (_, _, length) in enumerate(runs)
                   if 0 < i < len(runs) - 1 and length <= FLICKER_FRAMES)
    unmeasured = max((length for flag, _, length in runs_of(measured.tolist()) if not flag),
                     default=0)

    return {
        "frames": n,
        "unit": unit,
        "measured": {"frames": int(measured.sum()), "share": float(measured.sum() / n)},
        "offset": off,
        "modes": {m: {"frames": c, "share": c / n} for m, c in counts.most_common()},
        "transitions": {"count": len(runs) - 1, "per_100_frames": 100.0 * (len(runs) - 1) / n},
        "flickers": flickers,
        "longest_unmeasured_run": int(unmeasured),
    }


def figure(offset, measured, modes, result: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    offset = np.asarray(offset, dtype=np.float64)
    x = np.arange(len(modes))
    fig = plt.figure(figsize=(12, 4.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    strip = fig.add_subplot(grid[1, 0], sharex=ax)
    hx = fig.add_subplot(grid[0, 1], sharey=ax)

    shown = np.where(measured, offset, np.nan)
    ax.plot(x, shown, color="#1f77b4", linewidth=0.8)
    mean = result["offset"]["mean"]
    if mean is not None:
        ax.axhline(mean, color="black", linewidth=1, label=f"mean {mean:+.3f}")
        sd = result["offset"]["std"]
        ax.axhspan(mean - sd, mean + sd, color="#1f77b4", alpha=0.12, label=f"±1 std ({sd:.3f})")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylabel(f"offset ({result['unit']})")
    ax.set_title(title, fontsize=11, loc="left")
    ax.tick_params(labelbottom=False)
    ax.spines[["top", "right"]].set_visible(False)

    for mode, start, length in runs_of(modes):
        strip.axvspan(start - 0.5, start + length - 0.5, color=MODE_COLORS.get(mode, "#7f7f7f"),
                      linewidth=0)
    strip.set_yticks([])
    strip.set_xlim(-0.5, len(modes) - 0.5)
    strip.set_xlabel("frame")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS.get(m, "#7f7f7f"))
               for m in result["modes"]]
    strip.legend(handles, list(result["modes"]), fontsize=7, frameon=False, ncol=len(handles),
                 loc="upper center", bbox_to_anchor=(0.5, -0.9))

    v = offset[measured]
    if v.size:
        hx.hist(v, bins=30, orientation="horizontal", color="#1f77b4")
    hx.tick_params(labelleft=False)
    hx.set_xlabel("frames")
    hx.spines[["top", "right"]].set_visible(False)

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.stability", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run folder or CSV (default: newest)")
    p.add_argument("--normalized", action="store_true",
                   help="use the normalized offset even when cm are available")
    p.add_argument("--skip", type=int, default=WARMUP_FRAMES, help="drop the first N frames")
    p.add_argument("--out", help="output folder (default: next to the CSV)")
    args = p.parse_args(argv)

    try:
        path = common.find_csv(args.run, CSV_NAMES)
        offset, measured, modes, unit = load(Table(path, args.skip), args.normalized)
        res = analyze(offset, measured, modes, unit)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else path.parent
    common.write_json(out / "stability.json", res)
    fig = figure(offset, measured, modes, res, f"Still-scene stability: {path.parent.name}",
                 out / "stability.png")

    o, m = res["offset"], res["measured"]
    print(f"{path}  ({res['frames']} frames, offset in {unit})")
    print(f"  measured      {m['frames']} frames ({100 * m['share']:.1f}%)")
    print(f"  offset        mean {fmt(o['mean'], 3)}  std {fmt(o['std'], 3)}  "
          f"p5-p95 span {fmt(o['p5_p95_span'], 3)}")
    print("  modes         " + ", ".join(f"{k} {100 * v['share']:.1f}%" for k, v in res["modes"].items()))
    print(f"  transitions   {res['transitions']['count']} "
          f"({res['transitions']['per_100_frames']:.1f} per 100 frames), "
          f"{res['flickers']} flickers")
    print(f"  longest unmeasured run {res['longest_unmeasured_run']} frames")
    print(f"wrote {out / 'stability.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
