"""Navigation runs: what decided each frame, how well the lane was kept, and what each intersection did.

Purpose:
    navigation_linker and intersection_linker write nav.csv, one row per
    frame with the rule that decided it, the intersection stage, the lane
    state, the steering and the commands, the wheel speeds and the yaw.
    Reading it by hand answers one question at a time; this reduces a run
    to the numbers that say whether the robot drove well and where it
    didn't: how its time split between rules, how centered and how steady
    lane keeping was (weaving is steering that keeps changing sign), each
    intersection's stages, turn and heading, the 2 s after each crossing
    (where the robot has been seen to veer), whether the wheels turn alike
    at equal duty, and the frame-to-motor latency. Findings are listed in
    plain words against starting thresholds.

Main package:
    analyze(table) -> dict: run, rules, braking, lane_keeping,
        intersections, wheel_balance, latency, findings.
    report_lines(res): the printed report.
    nav_run.json, nav_run.png (rule / stage / lane strips over
    offset, steering, yaw and commands on one time axis).

Input:
    nav.csv (navigation_linker's NAV_FIELDS). For an intersection_linker
    run, give one maneuver's folder (runs/intersection_<time>/left).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table, runs_of, stats
from src.config import GYRO_BIAS_DPS
from src.navigation.lane_keeping import MAX_STEERING_ADJ
from src.navigation.navigation import (
    RULE_INTERSECTION, RULE_LANE_KEEPING, RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT,
)
from src.params import FPS, MODE_TWO_BOUNDARY

CSV_NAMES = ("nav.csv",)
LANE_VISION, LANE_STALE = "vision", "stale"
# Rules that act at an intersection: one unbroken run of them is one intersection
CROSSING_RULES = (RULE_INTERSECTION, RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT)
HELD_RULES = (RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT)

# Window after a crossing ends in which the robot has been seen to veer (2026-10-01 runs)
AFTER_S = 2.0
# Steering smaller than this isn't counted as a direction: sensor noise flips its sign
STEER_DEADBAND = 0.02
# Wheel balance: frames with equal commanded duty (within this) and both wheels turning
EQUAL_DUTY = 0.02
MIN_CPS = 100.0

# Finding thresholds. Starting values, not measured limits: adjust once a few
# good runs show what normal looks like
WEAVE_PER_S = 1.5           # steering sign changes per second of lane keeping
BIAS_CM, BIAS_NORM = 1.0, 0.07   # a mean offset this far off center is a bias (cm, or normalized)
STALE_SHARE = 0.05          # share of lane keeping on a stale lane
SATURATED_SHARE = 0.10      # share of lane keeping at full steering
VEER_RATIO = 1.5            # after a crossing, |offset| beyond this x the run's p95 is a veer
HEADING_TOLERANCE_DEG = 20.0
EXPECTED_DEG = {"left": -90.0, "right": 90.0, "straight": 0.0}
IMBALANCE_PCT = 5.0         # wheel speed difference at equal duty
LATENCY_BUDGET_MS = 1000.0 / FPS

PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

_CLI_HELP = """\
What decided each frame, how well the lane was kept, what each intersection
did and the 2 s after it, wheel balance and latency, from nav.csv. Writes
nav_run.png and nav_run.json next to the input.

Run from vision_stack/, venv active:
    python3 -m src.analysis.nav_run                          newest nav.csv
    python3 -m src.analysis.nav_run runs/nav_20261003_101500
    python3 -m src.analysis.nav_run runs/intersection_20261003_101500/left
"""


# =============================================================================
# Pieces
# =============================================================================

def frame_s(table: Table) -> np.ndarray:
    """Each frame's duration in seconds: its interval from the previous frame's t, the median for the first."""
    t = table.numeric("t") if table.has_values("t") else np.arange(len(table)) / FPS
    d = np.diff(t)
    first = float(np.nanmedian(d)) if d.size else 1.0 / FPS
    return np.concatenate(([first], d))


def _share_table(labels: list, dt: np.ndarray) -> dict:
    total = float(dt.sum()) or 1.0
    out = {}
    for label in dict.fromkeys(labels):
        mask = np.array([lb == label for lb in labels])
        out[label] = {"seconds": round(float(dt[mask].sum()), 2), "share": round(float(dt[mask].sum()) / total, 3),
                      "episodes": sum(1 for lb, _, _ in runs_of(labels) if lb == label)}
    return out


def rule_summary(table: Table, dt: np.ndarray) -> dict:
    """Time and episodes per deciding rule, and the commonest changes between rules."""
    rules = table.text("rule")
    pairs = Counter(f"{a} -> {b}" for (a, _, _), (b, _, _) in zip(runs_of(rules), runs_of(rules)[1:]))
    return {"by_rule": _share_table(rules, dt), "transitions": dict(pairs.most_common(8))}


def braking(table: Table, dt: np.ndarray) -> dict:
    brake = table.numeric("brake") == 1
    reasons = Counter(r for r, b in zip(table.text("reason"), brake) if b)
    return {"seconds": round(float(dt[brake].sum()), 2), "share": round(float(dt[brake].sum() / (dt.sum() or 1)), 3),
            "episodes": sum(1 for b, _, _ in runs_of(list(brake)) if b), "reasons": dict(reasons.most_common())}


def offset_column(table: Table, mask: np.ndarray) -> tuple[np.ndarray, str]:
    """The lane offset on the masked frames, in cm when the run has a cm scale, else normalized."""
    if table.has_values("lane_offset_cm") and not np.all(np.isnan(table.numeric("lane_offset_cm")[mask])):
        return table.numeric("lane_offset_cm"), "cm"
    return table.numeric("lane_offset"), "norm"


def weave_per_s(steer: np.ndarray, mask: np.ndarray, dt: np.ndarray) -> float:
    """Steering sign changes per second over the masked frames, counted only inside unbroken stretches of them."""
    changes, seconds = 0, float(dt[mask].sum())
    for on, start, length in runs_of(list(mask)):
        if not on:
            continue
        s = steer[start:start + length]
        signs = np.sign(s[np.abs(s) >= STEER_DEADBAND])
        changes += int(np.count_nonzero(np.diff(signs)))
    return round(changes / seconds, 2) if seconds > 0 else 0.0


def lane_keeping(table: Table, dt: np.ndarray) -> dict:
    """Lane keeping's frames: lane state, offset (vision frames), steering and weaving."""
    lk = np.array([r == RULE_LANE_KEEPING for r in table.text("rule")])
    if not lk.any():
        return {"seconds": 0.0}
    status = [s for s, m in zip(table.text("lane_status"), lk) if m]
    vision = lk & np.array([s == LANE_VISION for s in table.text("lane_status")])
    offset, unit = offset_column(table, vision)
    steer = table.numeric("steer")
    off = offset[vision]
    return {"seconds": round(float(dt[lk].sum()), 2),
            "lane_status": {k: v["share"] for k, v in _share_table(status, dt[lk]).items()},
            "offset_unit": unit,
            "offset": {**{k: stats(off)[k] for k in ("n", "mean", "std")},
                       "p95_abs": stats(np.abs(off))["p95"], "max_abs": stats(np.abs(off))["max"]},
            "steer_abs_mean": stats(np.abs(steer[lk]))["mean"],
            "saturated_share": round(float(np.mean(np.abs(steer[lk]) >= MAX_STEERING_ADJ - 1e-6)), 3),
            "weave_per_s": weave_per_s(steer, lk, dt)}


def after_crossings(table: Table, dt: np.ndarray, after_s: float = AFTER_S) -> np.ndarray:
    """Frames within after_s seconds after any intersection ends: where a veer would be."""
    t = table.numeric("t") if table.has_values("t") else np.cumsum(dt)
    mask = np.zeros(len(t), bool)
    for on, start, length in runs_of([r in CROSSING_RULES for r in table.text("rule")]):
        if on:
            end_t = t[start + length - 1]
            mask |= (t > end_t) & (t <= end_t + after_s)
    return mask


def baseline_p95(table: Table, after_s: float = AFTER_S) -> float | None:
    """
    p95 of |offset| on vision lane-keeping frames outside the windows after
    crossings: what normal lane keeping looks like, for judging a veer.
    """
    dt = frame_s(table)
    mask = (np.array([r == RULE_LANE_KEEPING for r in table.text("rule")])
            & np.array([s == LANE_VISION for s in table.text("lane_status")])
            & ~after_crossings(table, dt, after_s))
    offset, _ = offset_column(table, mask)
    return stats(np.abs(offset[mask]))["p95"]


def intersections(table: Table, dt: np.ndarray, after_s: float = AFTER_S,
                  gyro_bias_dps: float = GYRO_BIAS_DPS, baseline_p95: float | None = None) -> list[dict]:
    """
    One entry per intersection: each unbroken run of intersection, stop-sign
    or traffic-light frames.

    Each: step, maneuver, start_s, stage_s {to_line, turn, exit}, held_s,
    turn_end, turn_deg (the rule's heading when the turn ended), turned_deg
    (the gyro, net of gyro_bias_dps, over the whole crossing), and "after":
    the after_s seconds once it ends: lane-keeping share, offset on vision
    frames (mean, max |offset|), weaving, lane_back_s (until both lane lines
    on vision), and veer (max |offset| over VEER_RATIO x baseline_p95, the
    run's normal lane keeping: see baseline_p95()).
    """
    rules = table.text("rule")
    crossing = [r in CROSSING_RULES for r in rules]
    t = table.numeric("t") if table.has_values("t") else np.cumsum(dt)
    yaw = np.nan_to_num(table.numeric("yaw_rate"))
    stage, steps, man = table.text("stage"), table.text("step"), table.text("maneuver")
    turn_end, heading = table.text("turn_end"), table.numeric("heading_deg")
    status, mode, steer = table.text("lane_status"), table.text("lane_mode"), table.numeric("steer")
    out = []
    for on, start, length in runs_of(crossing):
        if not on:
            continue
        idx = range(start, start + length)
        stage_s = {s: round(float(sum(dt[i] for i in idx if stage[i] == s)), 2)
                   for s in ("to_line", "advance", "turn", "exit")}
        turn_rows = [i for i in idx if stage[i] == "turn" and not np.isnan(heading[i])]
        end_t = float(t[start + length - 1])
        after = np.array([end_t < t[i] <= end_t + after_s for i in range(len(t))])
        lk_after = after & np.array([r == RULE_LANE_KEEPING for r in rules])
        vis_after = lk_after & np.array([s == LANE_VISION for s in status])
        offset, _ = offset_column(table, vis_after)
        off = offset[vis_after]
        back = [i for i in range(start + length, len(t)) if t[i] - end_t <= after_s * 3
                and status[i] == LANE_VISION and mode[i] == MODE_TWO_BOUNDARY]
        max_abs = float(np.nanmax(np.abs(off))) if off.size and not np.all(np.isnan(off)) else None
        out.append({
            "step": next((steps[i] for i in reversed(idx) if steps[i]), ""),
            "maneuver": next((man[i] for i in reversed(idx) if man[i]), ""),
            "start_s": round(float(t[start]), 2), "seconds": round(float(sum(dt[i] for i in idx)), 2),
            "stage_s": stage_s,
            "held_s": round(float(sum(dt[i] for i in idx if rules[i] in HELD_RULES)), 2),
            "turn_end": next((turn_end[i] for i in idx if turn_end[i]), None),
            "turn_deg": round(float(heading[turn_rows[-1]]), 1) if turn_rows else None,
            "turned_deg": round(float(sum((yaw[i] - gyro_bias_dps) * dt[i] for i in idx)), 1),
            "after": {"seconds": round(float(dt[after].sum()), 2),
                      "lane_keeping_share": round(float(dt[lk_after].sum() / dt[after].sum()), 2) if after.any() else 0.0,
                      "offset_mean": stats(off)["mean"], "offset_max_abs": max_abs,
                      "weave_per_s": weave_per_s(steer, lk_after, dt),
                      "lane_back_s": round(float(t[back[0]] - end_t), 2) if back else None,
                      "veer": bool(max_abs is not None and baseline_p95 and max_abs > VEER_RATIO * baseline_p95)},
        })
    return out


def wheel_balance(table: Table) -> dict:
    """
    How alike the wheels turn at equal commanded duty, driving forward:
    imbalance_pct = mean of (left - right) / mean(left, right) x 100.
    + = the left wheel faster, which turns the robot right.
    """
    left, right = table.numeric("left_cps"), table.numeric("right_cps")
    cl, cr = table.numeric("cmd_left"), table.numeric("cmd_right")
    ok = ((table.numeric("brake") != 1) & (np.abs(cl - cr) < EQUAL_DUTY) & (cl > 0)
          & (left + right > 2 * MIN_CPS) & ~np.isnan(left) & ~np.isnan(right))
    if not ok.any():
        return {"frames": 0, "imbalance_pct": None}
    rel = (left[ok] - right[ok]) / ((left[ok] + right[ok]) / 2.0) * 100.0
    return {"frames": int(ok.sum()), "imbalance_pct": round(float(rel.mean()), 1),
            "left_cps_mean": round(float(left[ok].mean()), 1), "right_cps_mean": round(float(right[ok].mean()), 1)}


def latency(table: Table, dt: np.ndarray) -> dict:
    out = {}
    for col in ("latency_ms", "capture_ms", "phase2_ms", "phase3_ms", "nav_ms"):
        if table.has_values(col):
            s = stats(table.numeric(col))
            out[col] = {k: None if s[k] is None else round(s[k], 2) for k in ("p50", "p95", "max")}
    late = dt[1:] * 1000.0 > 1.5 * LATENCY_BUDGET_MS
    out["late_frames"] = int(late.sum())
    return out


# =============================================================================
# The whole run
# =============================================================================

def analyze(table: Table, after_s: float = AFTER_S, gyro_bias_dps: float = GYRO_BIAS_DPS) -> dict:
    missing = [c for c in ("rule", "lane_status", "steer", "brake") if not table.has(c)]
    if missing:
        raise ValueError(f"{table.path.name}: not a nav.csv (no {', '.join(missing)})")
    dt = frame_s(table)
    lk = lane_keeping(table, dt)
    p95 = baseline_p95(table, after_s)
    res = {"run": {"frames": len(table), "seconds": round(float(dt.sum()), 2),
                   "fps": round(len(table) / float(dt.sum()), 1) if dt.sum() else None,
                   "rejected": int(np.sum(np.array(table.text("reason")) == "contract"))},
           "rules": rule_summary(table, dt), "braking": braking(table, dt), "lane_keeping": lk,
           "intersections": intersections(table, dt, after_s, gyro_bias_dps, p95),
           "wheel_balance": wheel_balance(table), "latency": latency(table, dt)}
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    """Plain-words findings against the thresholds above; empty when nothing stands out."""
    out = []
    lk = res["lane_keeping"]
    if lk.get("seconds"):
        unit = lk["offset_unit"]
        bias = BIAS_CM if unit == "cm" else BIAS_NORM
        mean = lk["offset"]["mean"]
        if mean is not None and abs(mean) > bias:
            out.append(f"lane keeping runs {'right' if mean > 0 else 'left'} of center: mean offset "
                       f"{mean:+.2f} {unit} (threshold {bias})")
        if lk["weave_per_s"] > WEAVE_PER_S:
            out.append(f"lane keeping weaves: {lk['weave_per_s']} steering sign changes/s (threshold {WEAVE_PER_S}); "
                       "lower the steering gain or raise smoothing")
        if lk["lane_status"].get(LANE_STALE, 0) > STALE_SHARE:
            out.append(f"the lane was stale {100 * lk['lane_status'][LANE_STALE]:.0f}% of lane keeping "
                       f"(threshold {100 * STALE_SHARE:.0f}%): check the lane gates and lighting")
        if lk["saturated_share"] > SATURATED_SHARE:
            out.append(f"steering at its limit {100 * lk['saturated_share']:.0f}% of lane keeping: "
                       "the robot struggles to stay centered")
    for x in res["intersections"]:
        name = f"intersection {x['step'] or '?'}"
        if x["turn_end"] == "time limit":
            out.append(f"{name}: the turn ended on its time limit, not the gyro: check the IMU and gyro bias")
        exp = EXPECTED_DEG.get((x["maneuver"] or "").split()[-1] if x["maneuver"] else "")
        if exp is not None and abs(x["turned_deg"] - exp) > HEADING_TOLERANCE_DEG:
            out.append(f"{name}: turned {x['turned_deg']:+.0f} deg by the gyro, expected {exp:+.0f}")
        if x["after"]["veer"]:
            out.append(f"{name}: veered after the crossing, |offset| up to {x['after']['offset_max_abs']:.2f} "
                       f"{lk.get('offset_unit', '')} within {AFTER_S:.0f} s")
        if x["after"]["lane_back_s"] is None and x["after"]["seconds"]:
            out.append(f"{name}: both lane lines didn't come back on vision after the crossing")
    wb = res["wheel_balance"]
    if wb["imbalance_pct"] is not None and abs(wb["imbalance_pct"]) > IMBALANCE_PCT:
        side = "left" if wb["imbalance_pct"] > 0 else "right"
        out.append(f"the {side} wheel turns {abs(wb['imbalance_pct']):.0f}% faster at equal duty: "
                   f"the robot drifts {'right' if side == 'left' else 'left'} unless steering corrects it")
    lat = res["latency"].get("latency_ms")
    if lat and lat["p95"] is not None and lat["p95"] > LATENCY_BUDGET_MS:
        out.append(f"frame-to-motor latency p95 {lat['p95']:.0f} ms is over the {LATENCY_BUDGET_MS:.0f} ms frame budget")
    if res["run"]["rejected"]:
        out.append(f"{res['run']['rejected']} commands broke the contract and were braked")
    return out


def report_lines(res: dict) -> list[str]:
    r, lk, wb = res["run"], res["lane_keeping"], res["wheel_balance"]
    lines = [f"run           {r['frames']} frames, {r['seconds']} s, {r['fps']} FPS",
             "rules         " + ", ".join(f"{k} {100 * v['share']:.0f}% ({v['episodes']}x)"
                                         for k, v in res["rules"]["by_rule"].items()),
             f"braking       {res['braking']['seconds']} s in {res['braking']['episodes']} episodes"
             + (f": {', '.join(f'{k} {v}' for k, v in res['braking']['reasons'].items())}" if res["braking"]["reasons"] else "")]
    if lk.get("seconds"):
        o, u = lk["offset"], lk["offset_unit"]
        lines += [f"lane keeping  {lk['seconds']} s; lane " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in lk["lane_status"].items()),
                  f"  offset ({u}) mean {common.fmt(o['mean'])} std {common.fmt(o['std'])} p95 |x| {common.fmt(o['p95_abs'])}"
                  f"   steering |mean| {common.fmt(lk['steer_abs_mean'], 3)}, weave {lk['weave_per_s']}/s,"
                  f" at limit {100 * lk['saturated_share']:.0f}%"]
    for x in res["intersections"]:
        a = x["after"]
        lines.append(f"intersection  {x['step'] or '?':<12} at {x['start_s']:6.1f} s  to_line {x['stage_s']['to_line']} s,"
                     f" advance {x['stage_s']['advance']} s,"
                     f" turn {x['stage_s']['turn']} s, exit {x['stage_s']['exit']} s, held {x['held_s']} s;"
                     f" turn end {x['turn_end'] or '-'}, gyro {x['turned_deg']:+.0f} deg")
        lines.append(f"  after {AFTER_S:.0f} s: offset mean {common.fmt(a['offset_mean'])} max |x| {common.fmt(a['offset_max_abs'])},"
                     f" weave {a['weave_per_s']}/s, both lines back in {common.fmt(a['lane_back_s'])} s"
                     + ("  VEER" if a["veer"] else ""))
    lines.append(f"wheels        {wb['frames']} equal-duty frames" + (
        f": left {wb['left_cps_mean']} / right {wb['right_cps_mean']} cps, imbalance {wb['imbalance_pct']:+.1f}%"
        if wb["imbalance_pct"] is not None else " (none to compare)"))
    lat = res["latency"]
    if "latency_ms" in lat:
        lines.append(f"latency       frame to motor p50 {lat['latency_ms']['p50']} / p95 {lat['latency_ms']['p95']} /"
                     f" max {lat['latency_ms']['max']} ms; {lat['late_frames']} frames over 1.5x the budget")
    lines += ["", "findings"] + ([f"  - {f}" for f in res["findings"]] or ["  none: nothing past the thresholds"])
    return lines


# =============================================================================
# Figure and command line
# =============================================================================

def figure(table: Table, res: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    t = table.numeric("t") if table.has_values("t") else np.cumsum(frame_s(table))
    strips = [c for c in ("rule", "stage", "lane_status") if table.has(c)]
    lines = [("lane_offset_cm" if res["lane_keeping"].get("offset_unit") == "cm" else "lane_offset", "offset"),
             ("steer", "steering"), ("yaw_rate", "yaw deg/s")]
    fig, axes = plt.subplots(len(strips) + len(lines) + 1, 1, figsize=(13, 1.2 + 0.5 * len(strips) + 1.4 * (len(lines) + 1)),
                             sharex=True, gridspec_kw={"height_ratios": [0.5] * len(strips) + [1.4] * (len(lines) + 1)})
    for ax, col in zip(axes, strips):
        labels = table.text(col)
        colors = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(dict.fromkeys(labels)) if v}
        for v, start, length in runs_of(labels):
            if v:
                ax.axvspan(t[start], t[min(start + length, len(t) - 1)], color=colors[v], linewidth=0)
        ax.set_yticks([])
        ax.set_ylabel(col, rotation=0, ha="right", va="center", fontsize=8)
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], list(colors), fontsize=7,
                  frameon=False, ncol=min(len(colors), 6), loc="center left", bbox_to_anchor=(1.01, 0.5))
    for ax, (col, label) in zip(axes[len(strips):], lines):
        if table.has(col):
            ax.plot(t, table.numeric(col), lw=0.8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(label, fontsize=8)
    ax = axes[-1]
    for col, c in (("cmd_left", "#1f77b4"), ("cmd_right", "#d62728")):
        if table.has(col):
            ax.plot(t, table.numeric(col), lw=0.8, color=c, label=col)
    ax.set_ylabel("duty", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    for x in res["intersections"]:
        for a in axes[len(strips):]:
            a.axvspan(x["start_s"], x["start_s"] + x["seconds"], color="#ffdd88", alpha=0.35, linewidth=0)
            a.axvspan(x["start_s"] + x["seconds"], x["start_s"] + x["seconds"] + AFTER_S, color="#ffaaaa",
                      alpha=0.2, linewidth=0)
    axes[-1].set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    axes[-1].set_xlabel(f"time, s (yellow: intersection, red: the {AFTER_S:.0f} s after)")
    axes[0].set_title(title, fontsize=11, loc="left")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.nav_run", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run folder or nav.csv (default: newest nav.csv)")
    p.add_argument("--after-s", type=float, default=AFTER_S, help=f"window after each crossing (default {AFTER_S})")
    p.add_argument("--gyro-bias", type=float, default=GYRO_BIAS_DPS,
                   help=f"subtracted from yaw_rate for the turned angle (default {GYRO_BIAS_DPS}, config.GYRO_BIAS_DPS)")
    p.add_argument("--out", help="output folder (default: next to nav.csv)")
    args = p.parse_args(argv)
    try:
        path = common.find_csv(args.run, CSV_NAMES)
        table = Table(path)
        res = analyze(table, args.after_s, args.gyro_bias)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else path.parent
    common.write_json(out / "nav_run.json", res)
    fig = figure(table, res, f"Navigation run: {path.parent.name}", out / "nav_run.png")
    print(f"{path}\n" + "\n".join(report_lines(res)))
    print(f"wrote {out / 'nav_run.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
