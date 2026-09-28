#!/usr/bin/env python3
"""Stage timing figures: where each frame's time goes, against the frame budget.

Purpose:
    Reads per-frame stage timings and draws them against the 1000/FPS frame
    budget. Stages timed inside run_chain add up to total_ms; stages a run
    also times outside it (capture wait, HUD, recording) are drawn as their
    own segments. Whatever the loop's wall time still doesn't explain is
    drawn as "outside pipeline", so a slow loop shows where to look even
    before every part of it has a timer.

Main package:
    breakdown()           median / p95 / share of budget per stage, plus the
                          unaccounted and outside-pipeline remainders. Plain
                          numbers, no plotting, so tests and summaries use it.
    timing_budget.png     one bar across the budget: median per stage, the
                          remainders, then headroom. Pipeline and loop p95
                          are markers, since p95s of separate stages don't add.
    timing_per_frame.png  stages stacked per frame across the run, with the
                          loop time and the budget line, for spikes and drift.

Input:
    Any CSV with a *_ms column per stage: live_view's stages.csv, the
    stage_timing.csv test_stage_timing writes, or phase3_linker's p3.csv
    (capture, phase2, phase3). Stage columns are found by name, so a new
    timer shows up with no change here. Blank cells (a stage a run didn't
    time) are ignored, not read as zero. Whether total_ms already includes
    the external stages differs by writer (p3.csv's does, the others' don't)
    and is detected from the numbers.

matplotlib is optional, as for the test histograms: without it the figure
functions return None and breakdown() still works.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table, interval_ms
from src.params import FPS

# Pipeline order for the stages run_chain times, then anything timed around
# it. Unknown *_ms columns are appended after these in file order
STAGE_ORDER = ("capture", "preprocess", "roi", "geometry", "color",
               "lane_offset", "stop_line", "fusion", "package", "phase2", "phase3", "hud", "record")

# Timed outside run_chain, so not part of total_ms
EXTERNAL_STAGES = frozenset({"capture", "hud", "record"})

# *_ms columns that are not stage durations
NOT_STAGES = frozenset({"timestamp_ms", "total_ms", "interval_ms"})

# File names searched for, in order, when given a run folder
CSV_NAMES = ("stage_timing.csv", "stages.csv", "p3.csv")

UNACCOUNTED = "unaccounted"
OUTSIDE = "outside pipeline"
WARMUP_FRAMES = 5           # auto-exposure settling and first-call allocation

_CLI_HELP = """\
Draw where each frame's time goes, against the frame budget. Writes
timing_budget.png and timing_per_frame.png next to the input CSV.

Run from vision_stack/, venv active:
    python3 -m src.analysis.stage_timing                  newest run
    python3 -m src.analysis.stage_timing <folder or csv>  a specific run
Options take their value with "=" or a space: --fps=30, --skip 0.
"""


# =============================================================================
# Loading
# =============================================================================

def stage_names(columns) -> list:
    """
    Stage names from CSV column names: every *_ms column except the
    non-duration ones, known stages in pipeline order, unknown ones after.
    """
    found = [c[:-3] for c in columns if c.endswith("_ms") and c not in NOT_STAGES]
    known = [s for s in STAGE_ORDER if s in found]
    return known + [s for s in found if s not in known]


def read_timing_csv(path, skip: int = WARMUP_FRAMES) -> dict:
    """
    Read the timing columns of a stage CSV.

    Inputs:
        path: stages.csv, stage_timing.csv or p3.csv
        skip: leading frames to drop (warm-up)

    Outputs:
        {column: float64 array} for frame_id, timestamp_ms, every *_ms column,
        and interval_ms (see common.interval_ms). Blank cells are NaN.
    """
    table = Table(path, skip)
    cols = {c: table.numeric(c) for c in table.columns if c == "frame_id" or c.endswith("_ms")}
    iv = interval_ms(table)
    if iv is not None:
        cols["interval_ms"] = iv
    return cols


# =============================================================================
# Breakdown
# =============================================================================

def _stats(values: np.ndarray) -> dict:
    v = values[~np.isnan(values)]
    if v.size == 0:
        return {"median": None, "p95": None}
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}


def breakdown(cols: dict, budget_ms: float) -> dict:
    """
    Where the median frame's time goes.

    Inputs:
        cols: read_timing_csv() output
        budget_ms: frame budget, 1000 / target FPS

    Outputs:
        {
          "budget_ms", "frames",
          "stages":   {name: {"median", "p95", "share"}} in pipeline order,
          "total":    {"median", "p95"}  of run_chain's own wall time,
          "interval": {"median", "p95"} or None without loop timing,
          "unaccounted_ms": inside total_ms but in no recorded stage,
          "outside_ms":     loop time not explained by total_ms or the
                            external stages; None without loop timing,
          "segments": [(name, ms)] for the budget bar, in order,
          "headroom_ms": budget minus the segments, negative when over,
        }
        Segments are per-part medians, so they only approximate the median
        frame. p95s are reported per part and never added.
    """
    stages = stage_names(cols)
    internal = [s for s in stages if s not in EXTERNAL_STAGES]
    external = [s for s in stages if s in EXTERNAL_STAGES]

    def stack(names):
        if not names:
            return np.zeros(len(next(iter(cols.values()))))
        return np.nansum(np.vstack([cols[f"{n}_ms"] for n in names]), axis=0)

    total = cols["total_ms"] if "total_ms" in cols else stack(internal)
    # p3.csv's total_ms is capture + phase2 + phase3; the others' is the
    # chain alone. Take whichever reading leaves the smaller remainder
    with_ext = np.nanmedian(np.abs(total - stack(internal) - stack(external))) if external else np.inf
    without = np.nanmedian(np.abs(total - stack(internal)))
    total_includes_external = bool(external) and bool(with_ext < without)
    if total_includes_external:
        total = total - stack(external)
    unaccounted = np.clip(total - stack(internal), 0.0, None)

    per_stage = {}
    for s in stages:
        st = _stats(cols[f"{s}_ms"])
        st["share"] = None if st["median"] is None else st["median"] / budget_ms
        per_stage[s] = st

    interval = cols.get("interval_ms")
    has_interval = interval is not None and not np.all(np.isnan(interval))
    outside = (np.clip(interval - total - stack(external), 0.0, None)
               if has_interval else None)

    # Bar order follows the loop: capture, the chain and its unaccounted
    # remainder, the other external stages, then whatever is left
    def timed(names):
        return [(s, per_stage[s]["median"]) for s in names if per_stage[s]["median"] is not None]

    unacc_med = _stats(unaccounted)["median"] or 0.0
    outside_med = _stats(outside)["median"] if has_interval else None
    segments = (timed([s for s in external if s == "capture"])
                + timed(internal)
                + [(UNACCOUNTED, unacc_med)]
                + timed([s for s in external if s != "capture"]))
    if outside_med is not None:
        segments.append((OUTSIDE, outside_med))

    return {
        "budget_ms": budget_ms,
        "frames": int(len(total)),
        "stages": per_stage,
        "total": _stats(total),
        "total_includes_external": total_includes_external,
        "interval": _stats(interval) if has_interval else None,
        "unaccounted_ms": unacc_med,
        "outside_ms": outside_med,
        "segments": segments,
        "headroom_ms": budget_ms - sum(ms for _, ms in segments),
    }


# =============================================================================
# Figures
# =============================================================================

_pyplot = common.pyplot       # module-level name so tests can patch it


def _colors(plt, names) -> dict:
    cmap = plt.get_cmap("tab10")
    plain = [n for n in names if n not in (UNACCOUNTED, OUTSIDE)]
    out = {n: cmap(i % 10) for i, n in enumerate(plain)}
    out[UNACCOUNTED] = "#bdbdbd"
    out[OUTSIDE] = "#8c8c8c"
    return out


def budget_figure(bd: dict, title: str, out_path) -> Path | None:
    """The budget bar plus a median / p95 / share table. None without matplotlib."""
    plt = _pyplot()
    if plt is None:
        return None
    budget = bd["budget_ms"]
    segments = [(n, ms) for n, ms in bd["segments"] if ms > 0.0]
    colors = _colors(plt, list(bd["stages"]))     # same color per stage in bar and table

    marks = [bd["total"]["p95"] or 0.0]
    if bd["interval"]:
        marks.append(bd["interval"]["p95"] or 0.0)
    used = sum(ms for _, ms in segments)
    xmax = max(budget, used, *marks) * 1.08
    char_w = 0.0085 * xmax              # approx. width of one 8 pt character

    fig, (ax, tab) = plt.subplots(2, 1, figsize=(11, 5),
                                  gridspec_kw={"height_ratios": [1.2, 1]})
    left = 0.0
    for name, width in segments:
        ax.barh(0, width, left=left, height=0.5, color=colors[name], edgecolor="white",
                linewidth=1, hatch="//" if name == OUTSIDE else None)
        label = name.replace("_", " ")
        # Keep the label off the budget line when a segment crosses it
        visible = min(left + width, budget) - left if left < budget < left + width else width
        if visible >= (len(label) + 2) * char_w:
            ax.text(left + visible / 2, 0, label, ha="center", va="center", fontsize=8,
                    color="black" if name == UNACCOUNTED else "white")
        left += width
    if used < budget:
        ax.barh(0, budget - used, left=used, height=0.5, color="#f2f2f2",
                edgecolor="#cccccc", linewidth=1)
        ax.text(used + (budget - used) / 2, 0, "headroom", ha="center", va="center",
                fontsize=8, color="#666666")

    fps = 1000.0 / budget
    ax.axvline(budget, color="black", linewidth=1.5)
    ax.text(budget, 0.42, f"{budget:.0f} ms budget ({fps:g} FPS)", ha="right",
            va="bottom", fontsize=8)
    ax.axvline(marks[0], color="#d62728", linestyle="--", linewidth=1)
    ax.text(marks[0], 0.42, f" pipeline p95 {marks[0]:.1f}", ha="left", va="bottom",
            fontsize=8, color="#d62728")
    if bd["interval"]:
        ax.axvline(marks[1], color="#555555", linestyle=":", linewidth=1)
        ax.text(marks[1], -0.42, f" loop p95 {marks[1]:.1f}", ha="left", va="top",
                fontsize=8, color="#555555")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("ms per frame (median)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"{title}   ({bd['frames']} frames)", fontsize=11, loc="left")

    def row(name, st):
        share = "" if st["median"] is None else f"{100 * st['median'] / budget:.1f}%"
        fmt = lambda v: "" if v is None else f"{v:.2f}"
        return [name, fmt(st["median"]), fmt(st["p95"]), share]

    names = list(bd["stages"])
    rows = [row(n.replace("_", " "), bd["stages"][n]) for n in names]
    rows.append(row("pipeline total", bd["total"]))
    if bd["interval"]:
        rows.append(row("loop", bd["interval"]))
    tab.axis("off")
    table = tab.table(cellText=rows, colLabels=["stage", "median ms", "p95 ms", "of budget"],
                      cellLoc="center", colLoc="center", loc="upper center",
                      colWidths=[0.28, 0.16, 0.16, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.15)
    for i, n in enumerate(names, start=1):
        table[i, 0].set_facecolor((*colors[n][:3], 0.25))

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def per_frame_figure(cols: dict, budget_ms: float, title: str, out_path) -> Path | None:
    """Stages stacked per frame, with loop time and budget. None without matplotlib."""
    plt = _pyplot()
    if plt is None:
        return None
    stages = stage_names(cols)
    colors = _colors(plt, stages)
    n = len(next(iter(cols.values())))
    x = cols["frame_id"] if "frame_id" in cols else np.arange(n)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(x, [np.nan_to_num(cols[f"{s}_ms"]) for s in stages],
                 labels=[s.replace("_", " ") for s in stages],
                 colors=[colors[s] for s in stages], linewidth=0)
    top = budget_ms
    interval = cols.get("interval_ms")
    if interval is not None and not np.all(np.isnan(interval)):
        ax.plot(x, interval, color="#555555", linewidth=0.8, label="loop")
        top = max(top, float(np.nanpercentile(interval, 99)))
    ax.axhline(budget_ms, color="black", linewidth=1.2, label=f"{budget_ms:.0f} ms budget")
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(0, top * 1.15)
    ax.set_xlabel("frame")
    ax.set_ylabel("ms")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# =============================================================================
# CLI
# =============================================================================

def find_csv(arg, roots=common.SEARCH_ROOTS) -> Path:
    """A timing CSV from a file, a run folder, or (None) the newest run that has one."""
    return common.find_csv(arg, CSV_NAMES, roots)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.stage_timing",
                                description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run folder or CSV (default: newest under runs/)")
    p.add_argument("--fps", type=float, default=FPS, help=f"target rate (default {FPS})")
    p.add_argument("--skip", type=int, default=WARMUP_FRAMES,
                   help=f"drop the first N frames (default {WARMUP_FRAMES})")
    p.add_argument("--out", help="output folder (default: next to the CSV)")
    args = p.parse_args(argv)

    try:
        path = find_csv(args.run)
        cols = read_timing_csv(path, args.skip)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    budget = 1000.0 / args.fps
    bd = breakdown(cols, budget)
    out = Path(args.out) if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)
    title = path.parent.name

    written = [budget_figure(bd, f"Stage timing: {title}", out / "timing_budget.png"),
               per_frame_figure(cols, budget, f"Stage time per frame: {title}",
                                out / "timing_per_frame.png")]

    print(f"{path}  ({bd['frames']} frames, budget {budget:.1f} ms)")
    for name, ms in bd["segments"]:
        print(f"  {name:<18} {ms:7.2f} ms  {100 * ms / budget:5.1f}%")
    print(f"  {'headroom':<18} {bd['headroom_ms']:7.2f} ms")
    if written[0] is None:
        print("matplotlib not installed: no figures written")
    else:
        for w in written:
            print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
