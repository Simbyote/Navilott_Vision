"""State timelines: how long each Phase 3 state lasts and what it turns into.

Purpose:
    Navigation acts on lane_status, drive_state and stop_sign_detected, so
    how those change over a run matters as much as their per-frame values:
    how long the lane is held or stale before vision returns, whether a stop
    latches cleanly or chatters, which Phase 2 mode tends to precede a
    hold. This reads a per-frame log and reports, per field, each value's
    share of the run, how many episodes it had and how long they lasted,
    and every transition between values.

Main package:
    analyze()            per field: values, episodes, dwell, transitions
    state_timeline.png   one colored band per field across the run

Input:
    p3.csv by default (fields lane_status, drive_state, stop_sign_detected,
    p2_mode); --fields picks others from any CSV with one row per frame.
    Durations come from timestamp_ms when present, else frames at --fps.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table, runs_of
from src.params import FPS

CSV_NAMES = ("p3.csv",)
DEFAULT_FIELDS = ("lane_status", "drive_state", "stop_sign_detected", "p2_mode")
PALETTE = ("#2ca02c", "#ff7f0e", "#d62728", "#1f77b4", "#9467bd", "#8c564b",
           "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

_CLI_HELP = """\
How long each state lasts and what it changes into, per field. Writes
state_timeline.png and state_timeline.json next to the input CSV.

Run from vision_stack/, venv active:
    python3 -m src.analysis.state_timeline                  newest p3.csv
    python3 -m src.analysis.state_timeline <folder or csv>
    python3 -m src.analysis.state_timeline run.csv --fields mode
"""


def frame_ms(table: Table, fps: float) -> np.ndarray:
    """Each frame's duration: its interval to the previous frame, the median for the first."""
    iv = common.interval_ms(table)
    if iv is None or np.all(np.isnan(iv)):
        return np.full(len(table), 1000.0 / fps)
    fill = np.nanmedian(iv)
    return np.where(np.isnan(iv), fill, iv)


def analyze_field(values, durations_ms) -> dict:
    """
    Outputs:
        {
          "values": {value: {"frames", "share", "episodes",
                             "dwell_frames": {"median", "max"},
                             "dwell_ms": {"median", "max"}}},
          "transitions": {"count", "pairs": {"a -> b": n}},
        }
    """
    n = len(values)
    if n == 0:
        raise ValueError("no frames to analyze")
    durations_ms = np.asarray(durations_ms, dtype=np.float64)
    runs = runs_of(values)
    per_value = {}
    for value, count in Counter(values).most_common():
        mine = [(start, length) for v, start, length in runs if v == value]
        frames = np.array([length for _, length in mine], dtype=np.float64)
        ms = np.array([durations_ms[s:s + length].sum() for s, length in mine])
        per_value[value] = {
            "frames": count, "share": count / n, "episodes": len(mine),
            "dwell_frames": {"median": float(np.median(frames)), "max": int(frames.max())},
            "dwell_ms": {"median": float(np.median(ms)), "max": float(ms.max())},
        }
    pairs = Counter(f"{a[0]} -> {b[0]}" for a, b in zip(runs, runs[1:]))
    return {"values": per_value,
            "transitions": {"count": len(runs) - 1, "pairs": dict(pairs.most_common())}}


def analyze(table: Table, fields, fps: float = FPS) -> dict:
    present = [f for f in fields if table.has(f)]
    if not present:
        raise ValueError(f"{table.path.name}: none of {', '.join(fields)}")
    durations = frame_ms(table, fps)
    return {f: analyze_field(table.text(f), durations) for f in present}


def figure(table: Table, results: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    fields = list(results)
    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 0.9 + 0.75 * len(fields)),
                             sharex=True, squeeze=False)
    for ax, f in zip(axes[:, 0], fields):
        colors = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(results[f]["values"])}
        for v, start, length in runs_of(table.text(f)):
            ax.axvspan(start - 0.5, start + length - 0.5, color=colors[v], linewidth=0)
        ax.set_yticks([])
        ax.set_ylabel(f, rotation=0, ha="right", va="center", fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
        ax.legend(handles, list(colors), fontsize=7, frameon=False, ncol=min(len(colors), 6),
                  loc="center left", bbox_to_anchor=(1.01, 0.5))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("frame")
    axes[-1, 0].set_xlim(-0.5, len(table) - 0.5)
    axes[0, 0].set_title(title, fontsize=11, loc="left")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.state_timeline",
                                description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run folder or CSV (default: newest p3.csv)")
    p.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS),
                   help=f"columns to follow (default: {' '.join(DEFAULT_FIELDS)})")
    p.add_argument("--fps", type=float, default=FPS, help="frame rate when there are no timestamps")
    p.add_argument("--out", help="output folder (default: next to the CSV)")
    args = p.parse_args(argv)

    try:
        path = common.find_csv(args.run, CSV_NAMES)       # a file argument is used as given
        table = Table(path)
        res = analyze(table, args.fields, args.fps)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else path.parent
    common.write_json(out / "state_timeline.json", res)
    fig = figure(table, res, f"State timeline: {path.parent.name}", out / "state_timeline.png")

    print(f"{path}  ({len(table)} frames)")
    for f, r in res.items():
        print(f"{f}: {r['transitions']['count']} transitions")
        for v, s in r["values"].items():
            print(f"  {v:<22} {100 * s['share']:5.1f}%  {s['episodes']:>4} episodes  "
                  f"dwell median {s['dwell_ms']['median']:7.0f} ms  max {s['dwell_ms']['max']:7.0f} ms")
        for pair, count in list(r["transitions"]["pairs"].items())[:5]:
            print(f"    {pair}: {count}")
    print(f"wrote {out / 'state_timeline.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
