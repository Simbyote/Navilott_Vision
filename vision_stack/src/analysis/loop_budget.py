"""Loop budget: where a driving run's frame goes, capture to motor command and on to the next capture.

Purpose:
    The frame loop has 50 ms a frame. nav.csv times each phase (capture,
    Phase 2, Phase 3, navigation) and the time between frames; what neither
    explains is outside the timers (motor write, logging, the recorder). This
    draws the average frame phase by phase against the budget, each phase's
    spread on one log axis so the sub-millisecond ones show, Phase 2's time
    by lane mode, and the frame interval's histogram.

Main package:
    analyze(table, stages) -> dict; loop_budget.png and loop_budget.json beside nav.csv.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table
from src.params import FPS

CSV_NAME = "nav.csv"
BUDGET_MS = 1000.0 / FPS
# (column, label, what runs, colour); outside_ms is derived
PHASES = (("capture_ms", "Capture", "read the newest frame (waits if none is ready)", "#5F6B76"),
          ("phase2_ms", "Phase 2", "preprocess, regions, geometry, color, obstacle, fusion", "#1E63B5"),
          ("phase3_ms", "Phase 3", "lane filter, votes, heading, odometry", "#2A9A88"),
          ("nav_ms", "Navigation", "the rule in charge, steering, the command's contract", "#1D5F8C"),
          ("outside_ms", "Outside the timers", "motor write, logging, recorder, between frames", "#B26A00"))
# Phase 2's stages as summary.txt's [TIMING] names them, in run order
PHASE2_STAGES = ("preprocess", "roi", "geometry", "color", "lane_offset", "stop_line", "obstacle", "fusion",
                 "package", "branch_wait")
MODES = ("two_boundary", "left_only", "right_only", "single_uncalibrated", "none")
LABEL_MIN_MS = 6.0              # narrower bar segments are labelled above the bar

_CLI_HELP = """\
Where a navigation run's frames go: capture, Phase 2, Phase 3, navigation
and the time outside the timers, against the 50 ms frame. Writes
loop_budget.png and loop_budget.json next to nav.csv. Frame 0 (start-up) is
left out.

Run from vision_stack/:
    python3 -m src.analysis.loop_budget                       newest navigation run
    python3 -m src.analysis.loop_budget runs/nav_<time>
    make loop-budget RUN=runs/nav_<time>
"""


def parse_timing(text: str) -> dict:
    """summary.txt's [TIMING] block as {stage: (median, p95)}; {} without one."""
    if "[TIMING]" not in text:
        return {}
    block = re.split(r"\n\s*\n|\n\[(?!TIMING)", text.split("[TIMING]", 1)[1], maxsplit=1)[0]
    return {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"^\s*([\w.]+)\s+med\s+([\d.]+)\s+p95\s+([\d.]+)", block, re.M)}


def frames(table: Table) -> dict:
    """Per-frame columns without frame 0, with interval_ms (this capture to the next) and outside_ms."""
    t = table.numeric("t")
    cols = {k: table.numeric(k)[1:] for k, *_ in PHASES if k != "outside_ms" and table.has(k)}
    interval = np.diff(t)[1:] * 1000.0 if len(t) > 2 else np.array([])
    n = len(interval)
    cols = {k: v[:n] for k, v in cols.items()}
    timed = sum(np.nan_to_num(cols.get(k, np.zeros(n))) for k in ("capture_ms", "phase2_ms", "phase3_ms", "nav_ms"))
    cols["interval_ms"] = interval
    cols["outside_ms"] = interval - timed
    cols["lane_mode"] = table.text("lane_mode")[1:1 + n] if table.has("lane_mode") else [""] * n
    return cols


def analyze(table: Table, stages: dict | None = None) -> dict:
    """Means (they add up to the mean frame), medians, p95s and worsts per phase; Phase 2 by lane mode."""
    for c in ("t", "phase2_ms"):
        if not table.has(c):
            raise ValueError(f"{table.path} isn't a nav.csv (no {c})")
    f = frames(table)
    if len(f["interval_ms"]) < 2:
        raise ValueError(f"{table.path}: too few frames")
    iv = f["interval_ms"]
    phases = {}
    for key, label, what, _ in PHASES:
        v = f.get(key)
        if v is None:
            continue
        st = common.stats(v)
        phases[key] = {"label": label, "what": what, "mean": float(np.nanmean(v)), "median": st["p50"],
                       "p5": float(np.nanpercentile(v, 5)), "p95": st["p95"], "max": st["max"],
                       "share": st["p50"] / float(np.nanmedian(iv))}
    split = []
    if stages:
        named = [(s, stages[s][0]) for s in PHASE2_STAGES if s in stages]
        total = sum(w for _, w in named)
        if total > 0:
            scale = phases["phase2_ms"]["mean"] / total
            split = [(s, w * scale) for s, w in named]
    modes = {}
    for m in MODES:
        v = np.array([p for p, mode in zip(f["phase2_ms"], f["lane_mode"]) if mode == m])
        if v.size:
            modes[m] = {"frames": int(v.size), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
    res = {"frames": int(len(iv)), "fps": 1000.0 / float(np.nanmean(iv)), "interval": common.stats(iv),
           "over_budget_share": float(np.mean(iv > BUDGET_MS)), "mean_frame": float(np.nanmean(iv)),
           "phases": phases, "phase2_split": split, "phase2_by_mode": modes,
           "latency": common.stats(table.numeric("latency_ms")[1:]) if table.has("latency_ms") else None,
           "gc": int(np.sum(np.nan_to_num(table.numeric("gc_ms")) > 0)) if table.has("gc_ms") else None}
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    out = []
    med = res["interval"]["p50"]
    if med > BUDGET_MS:
        out.append(f"a frame takes {med:.1f} ms (median), {med - BUDGET_MS:.1f} ms over the {BUDGET_MS:.0f} ms budget: "
                   f"the loop runs at {1000 / med:.1f} FPS")
    p2 = res["phases"].get("phase2_ms")
    if p2 and p2["share"] > 0.6:
        out.append(f"Phase 2 is {100 * p2['share']:.0f}% of the frame ({p2['median']:.1f} ms median): the place to save time")
    out_ = res["phases"].get("outside_ms")
    small = sum(res["phases"][k]["median"] for k in ("phase3_ms", "nav_ms") if k in res["phases"])
    if out_ and small and out_["median"] > 2 * small:
        out.append(f"{out_['median']:.1f} ms a frame goes outside the timers, {out_['median'] / small:.0f}x Phase 3 "
                   "and navigation together: motor write, logging, the recorder")
    modes = res["phase2_by_mode"]
    if len(modes) > 1:
        lo, hi = min(modes.items(), key=lambda kv: kv[1]["median"]), max(modes.items(), key=lambda kv: kv[1]["median"])
        if hi[1]["median"] > 1.2 * lo[1]["median"]:
            out.append(f"Phase 2 takes {hi[1]['median']:.1f} ms on {hi[0]} frames against {lo[1]['median']:.1f} on {lo[0]}")
    return out


def figure(table: Table, res: dict, title: str, out_path):
    plt = common.pyplot()
    if plt is None:
        return None
    f = frames(table)
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.2, 1.1], hspace=0.6, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    x, small = 0.0, []
    for key, label, _, color in PHASES:
        if key not in res["phases"]:
            continue
        w = res["phases"][key]["mean"]
        short = "outside" if key == "outside_ms" else label
        parts = res["phase2_split"] if key == "phase2_ms" and res["phase2_split"] else [(short, w)]
        for i, (name, pw) in enumerate(parts):
            ax.barh(0, max(pw, 0), left=x, height=0.5, color=color, alpha=1.0 - 0.12 * (i % 4), edgecolor="white")
            if pw >= LABEL_MIN_MS:
                ax.text(x + pw / 2, 0, f"{name}\n{pw:.1f}", ha="center", va="center", fontsize=7, color="white")
            elif key != "phase2_ms":
                small.append((x + pw / 2, f"{label} {pw:.2f} ms" if pw < 1 else f"{label} {pw:.1f} ms"))
            x += max(pw, 0)
    for j, (xs, text) in enumerate(small):
        ax.annotate(text, (xs, 0.25), xytext=(xs, 0.55 + 0.28 * (j % 2)), fontsize=7, ha="center",
                    arrowprops=dict(arrowstyle="-", color="#5F6B76", lw=0.6))
    ax.axvline(BUDGET_MS, color="#B26A00", ls="--", lw=1)
    ax.text(BUDGET_MS, -0.42, f"{BUDGET_MS:.0f} ms", color="#B26A00", ha="center", va="top", fontsize=7)
    ax.text(x + 0.8, 0, f"{x:.1f} ms", va="center", fontsize=8)
    ax.set_xlim(0, max(x, BUDGET_MS) * 1.12)
    ax.set_ylim(-0.75, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("ms of the average frame, capture to the next capture (means: they add up)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    rows = [(k, *rest) for k, *rest in PHASES if k in res["phases"]][::-1]
    for i, (key, label, _, color) in enumerate(rows):
        p = res["phases"][key]
        lo = max(p["p5"], 0.01)
        ax.plot([lo, max(p["p95"], lo)], [i, i], color=color, lw=5, solid_capstyle="butt", alpha=0.85)
        ax.plot(max(p["median"], 0.01), i, "|", color="#1F2933", ms=12, mew=1.4)
        ax.plot(max(p["max"], 0.01), i, "o", color=color, ms=4)
        ax.text(lo / 1.3, i, f"{p['median']:.2f}" if p["median"] < 1 else f"{p['median']:.1f}", ha="right",
                va="center", fontsize=7)
    ax.axvline(BUDGET_MS, color="#B26A00", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.05, max(400, max(res["phases"][k]["max"] for k, *_ in rows) * 1.5))
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8)
    ax.set_xlabel("ms per frame, log (bar: p5-p95, tick: median, dot: worst)", fontsize=8)
    ax = fig.add_subplot(gs[2, 0])
    modes = [(m, s) for m, s in res["phase2_by_mode"].items()][::-1]
    for i, (m, s) in enumerate(modes):
        ax.barh(i, s["median"], height=0.55, color="#1E63B5")
        ax.plot([s["p95"]] * 2, [i - 0.3, i + 0.3], color="#1F2933", lw=1)
        ax.text(s["p95"] + 0.8, i, f"{s['median']:.1f} (p95 {s['p95']:.1f}) n={s['frames']}", va="center", fontsize=7)
    ax.axvline(BUDGET_MS, color="#B26A00", ls="--", lw=1)
    ax.set_yticks(range(len(modes)), [m for m, _ in modes], fontsize=7)
    ax.set_xlim(0, max([BUDGET_MS] + [s["p95"] for _, s in modes]) * 1.5)
    ax.set_xlabel("Phase 2 ms by lane mode (bar: median, tick: p95)", fontsize=8)
    ax = fig.add_subplot(gs[2, 1])
    iv = f["interval_ms"]
    top = max(BUDGET_MS * 1.6, float(np.nanpercentile(iv, 99)))
    ax.hist(np.clip(iv, 0, top), bins=60, color="#1E63B5")
    ax.axvline(BUDGET_MS, color="#B26A00", ls="--", lw=1)
    ax.text(BUDGET_MS * 1.02, ax.get_ylim()[1] * 0.9, f"{100 * res['over_budget_share']:.0f}% over",
            color="#B26A00", fontsize=7, va="top")
    ax.set_xlabel(f"time between frames, ms (clipped at {top:.0f})", fontsize=8)
    ax.set_ylabel("frames", fontsize=8)
    for a in fig.axes:
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    lines = [f"{res['frames']} frames at {res['fps']:.1f} FPS; time between frames median {res['interval']['p50']:.1f} ms "
             f"(p95 {res['interval']['p95']:.1f}), {100 * res['over_budget_share']:.0f}% over {BUDGET_MS:.0f} ms",
             f"{'phase':<20} {'mean':>6} {'median':>7} {'p95':>6} {'max':>7} {'of frame':>8}"]
    for key, p in res["phases"].items():
        lines.append(f"{p['label']:<20} {p['mean']:>6.2f} {p['median']:>7.2f} {p['p95']:>6.2f} {p['max']:>7.1f} "
                     f"{100 * p['share']:>7.1f}%")
    if res["latency"]:
        lines.append(f"frame in -> motor command: median {res['latency']['p50']:.1f} ms, p95 {res['latency']['p95']:.1f}")
    if res["gc"] is not None:
        lines.append(f"garbage collections during the run: {res['gc']}")
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.loop_budget", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="navigation run folder or nav.csv (default: the newest)")
    p.add_argument("--out", help="output folder (default: next to nav.csv)")
    args = p.parse_args(argv)
    try:
        path = common.find_csv(args.run, CSV_NAME)
        table = Table(path)
        summary = path.parent / "summary.txt"
        stages = parse_timing(summary.read_text()) if summary.is_file() else {}
        res = analyze(table, stages)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "loop_budget.json", res)
    fig = figure(table, res, f"Loop budget: {path.parent.name}", out / "loop_budget.png")
    print(path)
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'loop_budget.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
