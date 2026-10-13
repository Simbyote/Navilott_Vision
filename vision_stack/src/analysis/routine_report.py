"""Routine report: one routine run's verdict, criteria and trials, as figures.

Purpose:
    A routine (src/routines/) writes its verdict and trials as text and
    tables: summary.txt, trials.csv, results.json. This draws them, so a
    run can be read at a glance and put in a report: how far each criterion
    sat from its limit, what every trial measured, and the plot the routine's
    question asks for. That's a stop's gap against its band, a reported offset
    against the ruler, a read rate against the distance, or the pack's volts
    through each stage. Driving routines keep each attempt's
    navigation_linker folder (attempt_NN/nav.csv); each gets nav_run's
    report beside it.

Main package:
    analyze()            verdict, each criterion with its margin (+ = inside
                         the limit), each numeric field's stats, the attempts
    routine_report.png   the criteria as gauges (the pass region shaded, the
                         value marked), the routine's own plot where it has
                         one (KEY_PLOTS), then every numeric field per trial
    attempt_NN/nav_run.png, .json   nav_run on each attempt that has a nav.csv

Input:
    A routine folder, runs/routine_<name>_<time>/, with results.json (rows
    included). Without an argument, the newest one under runs/.
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

from src.analysis import common, nav_run
from src.analysis.common import Table, fmt

RESULTS = "results.json"
SKIP_FIELDS = frozenset({"trial", "t_s"})
MAX_FIELD_PLOTS = 12
STATE_SPREAD_CM = 0.8
GAP_IN = 0.65               # inches between the figure's rows: tick labels and the next title
KEY_GAP_IN = 1.0            # ... below the routine's own plot, which has an axis label too       # detect-range: each state's points this far apart at one gap
VERDICT_COLORS = {"PASS": "#2ca02c", "FAIL": "#d62728", "INCOMPLETE": "#ff7f0e", "RECORDED": "#1f77b4"}
PASS_COLOR, FAIL_COLOR, BAND_COLOR = "#2ca02c", "#d62728", "#2ca02c"
STATE_COLORS = {"red": "#d62728", "yellow": "#e6b400", "green": "#2ca02c", "off": "#7f7f7f",
                "present": "#d62728", "absent": "#7f7f7f"}

_CLI_HELP = """\
A routine run's verdict, criteria and trials as figures. Writes
routine_report.png and routine_report.json in the routine's folder, and
nav_run's report in each attempt_NN/ folder that has a nav.csv.

Run from vision_stack/, venv active:
    python3 -m src.analysis.routine_report runs/routine_<name>_<time>
    make routine-report RUN=runs/routine_<name>_<time>
"""


# =============================================================================
# Reading
# =============================================================================

def find_run(arg, roots=common.SEARCH_ROOTS) -> Path:
    """A routine folder: the one given (or a results.json in it), else the newest under roots."""
    if arg:
        p = Path(arg)
        folder = p.parent if p.is_file() else p
        if not (folder / RESULTS).is_file():
            raise FileNotFoundError(f"no {RESULTS} in {folder}: is it a routine folder?")
        return folder
    found = [f.parent for root in roots for f in Path(root).glob(f"routine_*/{RESULTS}")]
    if not found:
        raise FileNotFoundError(f"no routine_*/{RESULTS} under {', '.join(str(r) for r in roots)}")
    return max(found, key=lambda f: (f / RESULTS).stat().st_mtime)


def load(folder) -> tuple:
    """(result, rows) from a routine folder's results.json."""
    result = json.loads((Path(folder) / RESULTS).read_text())
    return result, list(result.get("rows", []))


# =============================================================================
# Numbers
# =============================================================================

def _number(v):
    """v as a float when it's a number (not a bool), else None."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def numeric_fields(rows: list) -> list:
    """The fields, in row order, whose filled values are all numbers and at least one is."""
    out = []
    for k in (rows[0] if rows else {}):
        if k in SKIP_FIELDS:
            continue
        vals = [r.get(k) for r in rows if r.get(k) not in (None, "")]
        if vals and all(_number(v) is not None for v in vals):
            out.append(k)
    return out


def varying(rows: list, fields: list) -> tuple:
    """(fields that change across the trials, {field: value} for those that don't)."""
    moving, fixed = [], {}
    for k in fields:
        vals = {_number(r.get(k)) for r in rows if _number(r.get(k)) is not None}
        if len(vals) > 1:
            moving.append(k)
        elif vals:
            fixed[k] = vals.pop()
    return moving, fixed


def margin(c: dict) -> float | None:
    """How far a criterion's value sits inside its limit (+) or outside it (-); None without a value."""
    v = c.get("value")
    if v is None:
        return None
    if c["op"] == "<=":
        return c["limit"] - v
    if c["op"] == ">=":
        return v - c["limit"]
    lo, hi = c["limit"]
    return min(v - lo, hi - v)


def analyze(result: dict, rows: list, folder=None) -> dict:
    """
    Outputs:
        {
          "routine", "verdict", "requirement", "trials_kept", "trials_planned",
          "criteria": [{"name", "value", "op", "limit", "unit", "passed", "margin"}],
          "fields": {field: common.stats() over the trials},
          "attempts": [attempt folder names with a nav.csv],
        }
    """
    criteria = [{k: c.get(k) for k in ("name", "value", "op", "limit", "unit", "passed")}
                | {"margin": None if margin(c) is None else round(margin(c), 4)}
                for c in result.get("criteria", [])]
    fields = {k: common.stats([_number(r.get(k)) if _number(r.get(k)) is not None else np.nan for r in rows])
              for k in numeric_fields(rows)}
    attempts = [] if folder is None else [p.parent.name for p in sorted(Path(folder).glob("attempt_*/nav.csv"))]
    return {"routine": result.get("routine"), "verdict": result.get("verdict"),
            "requirement": result.get("requirement"), "trials_kept": result.get("trials_kept"),
            "trials_planned": result.get("trials_planned"), "criteria": criteria, "fields": fields,
            "attempts": attempts}


# =============================================================================
# The routines' own plots
# =============================================================================

def _criterion(result: dict, name_start: str):
    return next((c for c in result.get("criteria", []) if c["name"].startswith(name_start)), None)


def _col(rows, k):
    return np.array([_number(r.get(k)) if _number(r.get(k)) is not None else np.nan for r in rows])


def _plot_tape_check(ax, rows, result, folder):
    y = _col(rows, "reading_cm")
    ax.plot(range(1, len(y) + 1), y, "o-", color="#1f77b4")
    if np.any(~np.isnan(y)):
        ax.axhline(np.nanmean(y), color="black", linewidth=1, label=f"mean {np.nanmean(y):.2f} cm")
        ax.set_title(f"readings of one distance: spread {np.nanmax(y) - np.nanmin(y):.2f} cm", loc="left")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("reading")
    ax.set_ylabel("cm")


def _plot_stop_distance(ax, rows, result, folder):
    gap, volts = _col(rows, "gap_cm"), _col(rows, "battery_v")
    c = _criterion(result, "mean gap")
    if c:
        ax.axhspan(*c["limit"], color=BAND_COLOR, alpha=0.12, label=f"mean gap band {c['limit'][0]:g}..{c['limit'][1]:g} cm")
    ax.axhline(0, color="black", linewidth=1, label="the line")
    x = np.arange(1, len(gap) + 1)
    if np.any(~np.isnan(volts)):
        sc = ax.scatter(x, gap, c=volts, cmap="viridis", zorder=3)
        ax.figure.colorbar(sc, ax=ax, label="battery V")
    else:
        ax.plot(x, gap, "o", color="#1f77b4", zorder=3)
    if np.any(~np.isnan(gap)):
        ax.axhline(np.nanmean(gap), color="#1f77b4", linestyle="--", linewidth=1, label=f"mean {np.nanmean(gap):.2f} cm")
    ax.set_title("gap from the bumper to the line, per stop (+ = stopped before it)", loc="left")
    ax.set_xlabel("trial")
    ax.set_ylabel("gap (cm)")
    ax.legend(fontsize=8, frameon=False)


def _plot_lane_offset(ax, rows, result, folder):
    true, rep, sd = _col(rows, "measured_cm"), _col(rows, "reported_cm"), _col(rows, "sd_cm")
    c = _criterion(result, "worst")
    tol = c["limit"] if c else 2.0
    ok = ~np.isnan(true)
    span = np.nanmax(np.abs(np.concatenate([true[ok], rep[ok & ~np.isnan(rep)], [tol]]))) * 1.2 if ok.any() else 5.0
    line = np.array([-span, span])
    ax.fill_between(line, line - tol, line + tol, color=BAND_COLOR, alpha=0.12, label=f"±{tol:g} cm")
    ax.plot(line, line, color="black", linewidth=1, label="reported = ruler")
    ax.errorbar(true, rep, yerr=np.nan_to_num(sd), fmt="o", color="#1f77b4", capsize=3, label="trial (±1 sd)")
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("reported offset against the ruler (+ = right of center)", loc="left")
    ax.set_xlabel("ruler (cm)")
    ax.set_ylabel("reported (cm)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def _plot_detect_range(ax, rows, result, folder):
    states = list(dict.fromkeys(r.get("state") for r in rows))
    for k, state in enumerate(states):
        rs = [r for r in rows if r.get("state") == state]
        # each state a little to the side, so equal reads don't hide each other
        gap = _col(rs, "gap_cm") + (k - (len(states) - 1) / 2) * STATE_SPREAD_CM
        seen = _col(rs, "seen_pct")
        ok = np.array([r.get("result") == "ok" for r in rs])
        color = STATE_COLORS.get(str(state), "#1f77b4")
        order = np.argsort(gap)
        ax.plot(gap[order], seen[order], "-", color=color, linewidth=1, alpha=0.6)
        ax.scatter(gap[ok], seen[ok], color=color, zorder=3, label=f"{state}")
        ax.scatter(gap[~ok], seen[~ok], s=90, facecolors="white", edgecolors=color, linewidths=2, zorder=4)
    need = (result.get("options") or {}).get("need_cm")
    if _number(need) is None:
        try:
            need = float(str(need))
        except ValueError:
            need = None
    if need is not None:
        ax.axvline(float(need), color="black", linestyle="--", linewidth=1, label=f"needed from {float(need):g} cm")
    rng = (result.get("state") or {}).get("range_cm")
    title = "frames with the target seen, per gap (hollow = not read right)"
    ax.set_title(title + ("" if rng is None else f"; range {rng:g} cm"), loc="left")
    ax.set_xlabel("bumper to the stop line (cm)")
    ax.set_ylabel("seen (% of frames)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, frameon=False)


def _plot_recover_offset(ax, rows, result, folder):
    start, rec_s = _col(rows, "planned_cm"), _col(rows, "recover_s")
    ok = np.array([bool(r.get("recovered")) for r in rows])
    ax.scatter(start[ok], rec_s[ok], color=PASS_COLOR, zorder=3, label="recovered")
    ax.scatter(start[~ok], np.zeros(int((~ok).sum())), marker="x", s=60, color=FAIL_COLOR, zorder=3,
               label="not recovered (drawn at 0)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("time to steer back, per start offset", loc="left")
    ax.set_xlabel("start offset (cm, + = right)")
    ax.set_ylabel("recover time (s)")
    ax.legend(fontsize=8, frameon=False, loc="upper center")


def _plot_power_profile(ax, rows, result, folder):
    c = _criterion(result, "lowest pack volts")
    samples = None if folder is None else Path(folder) / "samples.csv"
    if samples is not None and samples.is_file():
        t = Table(samples)
        stages, tt, v = t.text("stage"), t.numeric("t_s"), t.numeric("battery_v")
        offset, ticks = 0.0, []
        for stage in dict.fromkeys(stages):
            m = np.array([s == stage for s in stages])
            x = tt[m] + offset
            ax.plot(x, v[m], linewidth=1.2, label=stage)
            if x.size:
                ticks.append(((x[0] + x[-1]) / 2, stage))
                offset = x[-1] + 2.0
        ax.set_xlabel("s (stages end to end)")
        ax.set_title("pack volts through each stage", loc="left")
    else:
        names = [str(r.get("stage")) for r in rows]
        x = np.arange(len(names))
        ax.bar(x, _col(rows, "v_mean"), color="#1f77b4", alpha=0.6, label="mean")
        ax.plot(x, _col(rows, "v_min"), "v", color="black", label="lowest")
        ax.set_xticks(x, names)
        ax.set_title("pack volts per stage", loc="left")
    if c:
        ax.axhline(c["limit"], color=FAIL_COLOR, linestyle="--", linewidth=1, label=f"warning {c['limit']:g} V")
    ax.set_ylabel("battery (V)")
    ax.legend(fontsize=8, frameon=False)


def _plot_imu_check(ax, rows, result, folder):
    turns = [r for r in rows if r.get("part") == "turn"]
    err = _col(turns, "error_deg")
    x = np.arange(len(turns))
    c = _criterion(result, "turns off 90")
    m = re.search(r"over ([0-9.]+) deg", c["name"]) if c else None
    tol = float(m.group(1)) if m else None
    if tol is not None:
        ax.axhspan(-tol, tol, color=BAND_COLOR, alpha=0.12, label=f"±{tol:g} deg")
        ax.legend(fontsize=8, frameon=False)
    ax.bar(x, err, color=[FAIL_COLOR if tol is not None and abs(e) > tol else PASS_COLOR
                          for e in np.nan_to_num(err)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{i + 1} {r.get('direction', '')}" for i, r in enumerate(turns)])
    ax.set_title("gyro heading minus 90 deg, per hand turn", loc="left")
    ax.set_ylabel("error (deg)")


def _plot_obstacle_check(ax, rows, result, folder):
    present = [r for r in rows if r.get("state") == "present"]
    gap, rep = _col(present, "gap_cm"), _col(present, "reported_cm")
    ok = np.array([r.get("result") == "ok" for r in present], bool)
    clipped = np.array([_number(r.get("clipped_pct")) not in (None, 0.0) and np.isnan(rep[i])
                        for i, r in enumerate(present)], bool)
    missed = np.array([r.get("result") == "missed" for r in present], bool)
    c = _criterion(result, "worst distance error")
    tol = c["limit"] if c else 3.0
    hi = float(np.nanmax(np.concatenate([gap, rep, [20.0]]))) * 1.15 if len(gap) else 40.0
    line = np.array([0.0, hi])
    ax.fill_between(line, line - tol, line + tol, color=BAND_COLOR, alpha=0.12, label=f"±{tol:g} cm")
    ax.plot(line, line, color="black", linewidth=1, label="reported = tape")
    m = ~np.isnan(rep)
    ax.scatter(gap[m & ok], rep[m & ok], color=PASS_COLOR, zorder=3, label="seen, measured")
    ax.scatter(gap[m & ~ok], rep[m & ~ok], color=FAIL_COLOR, zorder=3, label="distance off")
    ax.scatter(gap[clipped], np.zeros(int(clipped.sum())), marker="v", color="#7f7f7f", zorder=3,
               label="below the view (drawn at 0)")
    ax.scatter(gap[missed], np.zeros(int(missed.sum())), marker="x", s=60, color=FAIL_COLOR, zorder=3,
               label="not seen (drawn at 0)")
    b = _criterion(result, "brake decisions")
    m2 = re.search(r"from ([0-9.]+) cm", b["name"]) if b else None
    if m2:
        ax.axvline(float(m2.group(1)), color="#555555", linestyle="--", linewidth=1, label="the rule brakes nearer")
    falses = sum(1 for r in rows if r.get("state") == "clear" and r.get("result") == "wrong")
    clears = sum(1 for r in rows if r.get("state") == "clear")
    ax.set_title(f"reported distance against the tape; clear lane: {falses} false alarms in {clears}", loc="left")
    ax.set_xlim(0, hi)
    ax.set_ylim(-2, hi)
    ax.set_xlabel("tape: front edge to the object (cm)")
    ax.set_ylabel("reported (cm)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def _plot_obstacle_stop(ax, rows, result, folder):
    objs = [r for r in rows if r.get("state") == "object"]
    gap = _col(objs, "gap_cm")
    c = _criterion(result, "gaps outside")
    m = re.search(r"outside ([0-9.]+)\.\.([0-9.]+) cm", c["name"]) if c else None
    if m:
        ax.axhspan(float(m.group(1)), float(m.group(2)), color=BAND_COLOR, alpha=0.12,
                   label=f"{m.group(1)}..{m.group(2)} cm")
    ax.axhline(0, color="black", linewidth=1, label="the object")
    stopped = np.array([bool(r.get("stopped")) for r in objs], bool)
    x = np.arange(1, len(objs) + 1)
    ax.scatter(x[stopped], gap[stopped], color=PASS_COLOR, zorder=3, label="stopped")
    ax.scatter(x[~stopped], np.nan_to_num(gap[~stopped]), marker="x", s=60, color=FAIL_COLOR, zorder=3,
               label="didn't stop")
    falses = sum(int(_number(r.get("brakes")) or 0) for r in rows if r.get("state") == "clear")
    ax.set_xticks(x)
    ax.set_title(f"gap from the front edge to the object, per stop; clear lane: {falses} false brakes", loc="left")
    ax.set_xlabel("object trial")
    ax.set_ylabel("gap (cm)")
    ax.legend(fontsize=8, frameon=False)


def _plot_stop_sign(ax, rows, result, folder):
    signs = [r for r in rows if r.get("sign") == "present"]
    hold = _col(signs, "hold_s")
    c = _criterion(result, "worst hold off")
    m = re.search(r"off ([0-9.]+) s", c["name"]) if c else None
    if m:
        target, tol = float(m.group(1)), c["limit"]
        ax.axhspan(target - tol, target + tol, color=BAND_COLOR, alpha=0.12, label=f"{target:g} ± {tol:g} s")
    x = np.arange(1, len(signs) + 1)
    stopped = np.array([(_number(r.get("stops")) or 0) >= 1 for r in signs], bool)
    ax.scatter(x[stopped], hold[stopped], s=50, color=PASS_COLOR, zorder=3, label="held")
    ax.set_xticks(x)
    ax.set_ylim(-0.2, max(3.0, float(np.nanmax(hold)) + 0.5) if np.any(~np.isnan(hold)) else 3.0)
    ax.scatter(x[~stopped], np.zeros(int((~stopped).sum())), marker="x", s=60, color=FAIL_COLOR, zorder=3,
               label="didn't stop")
    wrong = sum(int(_number(r.get("stops")) or 0) for r in rows if r.get("sign") == "absent")
    nones = sum(1 for r in rows if r.get("sign") == "absent")
    ax.set_title(f"hold at the line, per sign trial; no sign: {wrong} stops in {nones} trials", loc="left")
    ax.set_xlabel("sign trial")
    ax.set_ylabel("hold (s)")
    ax.legend(fontsize=8, frameon=False)


# Routine name -> its own plot: the one its question asks for
KEY_PLOTS = {
    "tape-check": _plot_tape_check,
    "stop-distance": _plot_stop_distance,
    "lane-offset": _plot_lane_offset,
    "detect-range": _plot_detect_range,
    "recover-offset": _plot_recover_offset,
    "power-profile": _plot_power_profile,
    "imu-check": _plot_imu_check,
    "obstacle-check": _plot_obstacle_check,
    "obstacle-stop": _plot_obstacle_stop,
    "stop-sign": _plot_stop_sign,
}


# =============================================================================
# The figure
# =============================================================================

def _gauge(ax, c: dict) -> None:
    """One criterion: its pass region shaded, the limit(s) drawn, the value marked."""
    v, op, lim = c.get("value"), c["op"], c["limit"]
    edges = list(lim) if op == "within" else [lim]
    pts = edges + ([v] if v is not None else [])
    lo, hi = min(pts), max(pts)
    pad = max((hi - lo) * 0.25, abs(hi) * 0.1, 1.0)
    floor = 0.0 if lo >= 0 else None          # counts and sizes: no negative axis
    lo, hi = lo - pad, hi + pad
    if floor is not None:
        lo = max(lo, floor - 0.05 * (hi - floor))
    if op == "<=":
        ax.axvspan(lo, lim, color=BAND_COLOR, alpha=0.15)
    elif op == ">=":
        ax.axvspan(lim, hi, color=BAND_COLOR, alpha=0.15)
    else:
        ax.axvspan(lim[0], lim[1], color=BAND_COLOR, alpha=0.15)
    for e in edges:
        ax.axvline(e, color="black", linewidth=1)
    if v is not None:
        ax.plot([v], [0], "o", markersize=9, color=PASS_COLOR if c["passed"] else FAIL_COLOR, zorder=3)
    else:
        ax.text((lo + hi) / 2, 0, "no value", ha="center", va="center", color=FAIL_COLOR, fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)
    mark = "PASS" if c["passed"] else "FAIL"
    ax.set_title(f"{mark}  {c.get('text') or c['name']}", fontsize=8, loc="left",
                 color=PASS_COLOR if c["passed"] else FAIL_COLOR)


def figure(result: dict, rows: list, res: dict, folder, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    criteria = result.get("criteria", [])
    key = KEY_PLOTS.get(result.get("routine"))
    fields, fixed = varying(rows, numeric_fields(rows))
    fields = fields[:MAX_FIELD_PLOTS]
    cols = 3
    field_rows = -(-len(fields) // cols)
    # Each block's height in inches; a spacer row of GAP_IN after each holds its
    # tick labels and the next one's title (an even hspace would shrink beside the big plot)
    blocks = ([(0.8, GAP_IN)] + [(0.7, GAP_IN)] * len(criteria) + ([(4.0, KEY_GAP_IN)] if key and rows else [])
              + [(1.9, GAP_IN)] * field_rows)
    heights = [h for pair in blocks for h in pair]
    fig = plt.figure(figsize=(11, sum(heights)))
    grid = fig.add_gridspec(len(heights), cols, height_ratios=heights, wspace=0.3, hspace=0)

    head = fig.add_subplot(grid[0, :])
    head.axis("off")
    verdict = result.get("verdict", "")
    cond = (result.get("conditions") or {}).get("start") or {}
    head.text(0, 1, f"{result.get('title') or result.get('routine')}: {verdict}", fontsize=13, va="top",
              color=VERDICT_COLORS.get(verdict, "black"), transform=head.transAxes)
    sub = [f"verifies {result['requirement']}" if result.get("requirement") else "",
           f"{result.get('trials_kept')} of {result.get('trials_planned')} trials",
           f"commit {cond.get('commit') or '--'}" + (" (uncommitted changes)" if cond.get("uncommitted_changes") else ""),
           str(cond.get("time") or ""), f"tester {cond.get('tester') or '--'}"]
    head.text(0, 0.1, textwrap.fill("  ·  ".join(x for x in sub if x), 150), fontsize=8, color="#555555",
              va="bottom", transform=head.transAxes)

    row = 2                                 # grid rows: block, spacer, block, ...
    for c in criteria:
        _gauge(fig.add_subplot(grid[row, :]), c)
        row += 2
    if key and rows:
        ax = fig.add_subplot(grid[row, :])
        key(ax, rows, result, folder)
        ax.spines[["top"]].set_visible(False)
        row += 2
    for i, k in enumerate(fields):
        ax = fig.add_subplot(grid[row + 2 * (i // cols), i % cols])
        y = _col(rows, k)
        x = [r.get("trial", j + 1) for j, r in enumerate(rows)]
        ax.plot(x, y, "o-", color="#1f77b4", markersize=4, linewidth=1)
        if np.any(~np.isnan(y)):
            ax.axhline(np.nanmean(y), color="black", linewidth=0.8, linestyle="--")
        ax.set_title(k, fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    if fixed:
        what = "the trial's values" if len(rows) == 1 else "the same in every trial"
        fig.text(0.06, 0.0, textwrap.fill(f"{what}: " + ", ".join(f"{k} {v:g}" for k, v in fixed.items()), 150),
                 fontsize=8, color="#555555", va="top")
    if not criteria and not fields and not key:
        head.text(0, -0.5, "no trials kept", transform=head.transAxes)

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def attempt_reports(folder) -> list:
    """nav_run's json and figure in each attempt_NN/ with a nav.csv; [(attempt, png or None, error or None)]."""
    out = []
    for nav in sorted(Path(folder).glob("attempt_*/nav.csv")):
        try:
            table = Table(nav)
            res = nav_run.analyze(table)
        except ValueError as e:
            out.append((nav.parent.name, None, str(e)))
            continue
        common.write_json(nav.parent / "nav_run.json", res)
        png = nav_run.figure(table, res, f"Navigation run: {Path(folder).name}/{nav.parent.name}",
                             nav.parent / "nav_run.png")
        out.append((nav.parent.name, png, None))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.routine_report", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="routine folder (default: the newest runs/routine_*)")
    p.add_argument("--no-attempts", action="store_true", help="skip nav_run on the attempt_NN folders")
    args = p.parse_args(argv)
    try:
        folder = find_run(args.run)
        result, rows = load(folder)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    res = analyze(result, rows, folder)
    common.write_json(folder / "routine_report.json", res)
    fig = figure(result, rows, res, folder, folder / "routine_report.png")

    print(f"{folder}  ({res['routine']}: {res['verdict']}, {res['trials_kept']} of {res['trials_planned']} trials)")
    for c in res["criteria"]:
        side = "" if c["margin"] is None else ("inside by " if c["margin"] >= 0 else "outside by ") + fmt(abs(c["margin"]), 3)
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {fmt(c['value'], 3) or '--'} {c['unit'] or ''}"
              f"  {side}".rstrip())
    if not res["criteria"]:
        print("  no criteria: a characterization")
    for k, s in res["fields"].items():
        print(f"  {k:<20} mean {fmt(s['mean'], 3):>9}  min {fmt(s['min'], 3):>9}  max {fmt(s['max'], 3):>9}")
    print(f"wrote {folder / 'routine_report.json'}" + (f"\nwrote {fig}" if fig else "\n(no matplotlib: no figure)"))
    if not args.no_attempts:
        for name, png, err in attempt_reports(folder):
            print(f"  {name}: " + (f"ERROR {err}" if err else f"wrote {png or (name + '/nav_run.json')}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
