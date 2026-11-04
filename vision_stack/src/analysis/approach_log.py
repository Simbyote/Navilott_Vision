"""Approach log: a driving run's nav.csv cut into one row per stop-line approach, for the dynamic side of verification.

Purpose:
    The bench tests (TP-04 to TP-07) judge one parked pose at a time. While
    driving, the robot sees each line, sign and light over a whole approach,
    with the 2-of-3 vote and the sign memory behind it. This turns a course or
    figure-eight run into the same quantities per approach (when the sign was
    first seen, how many frames saw the line, what the light read, how long it
    held) and, by distance to the line, the shares the bench measures per gap,
    so the two can sit side by side in the verification workbook.

    Distance to the line comes from odometry, counted back from where the
    robot reached the line (the stop-line tracker's "reached" frame, with
    line_left_cm of the line still ahead): est_cm = line_left_cm at reach +
    the cm driven from that frame to reach. It is the bench's reference (front
    edge to the line's near edge). A taped stop gap entered in the workbook
    corrects the constant; the slope is the encoders'.

Main package:
    analyze(nav_csv, sign_at, light_at) -> dict with approaches, bins, run;
    approaches.csv, dynamic_bins.csv and run_summary.csv in --out.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

from src.analysis import common

CROSSING, IDLE = "crossing", "idle"
RULE_INTERSECTION = "intersection"
HOLD_REASONS = ("stop_sign_stopping", "stop_sign_hold")
RED_REASON, OBSTACLE_REASON = "red_light", "obstacle"
STATE_RANK = {"go": 0, "caution": 1, "stop": 2}     # Phase 3 drive_state: caution = yellow, stop = red
REACH_CM = 9.0                                      # stop_line.REACH_CM, when line_left_cm isn't logged
BIN_CM, BIN_MAX_CM = 5.0, 50.0
LAP_LEN = 14                                        # figure_eight.ROUTE_LAP; --lap-len for another route

APPROACH_FIELDS = ("run", "approach", "lap", "position", "step", "maneuver", "t_start", "t_reach", "seconds",
                   "frames", "fps", "sign_expected", "sign_frames", "sign_first_est_cm", "sign_last_est_cm",
                   "line_frames", "line_first_cm", "line_first_est_cm", "line_vs_odom_cm", "left_at_reach_cm",
                   "light_expected", "light_read", "red_frames", "yellow_frames",
                   "stopped_for", "hold_s", "red_wait_s", "obstacle_s",
                   "lane_max_cm", "lane_lost_frames", "latency_p95_ms", "battery_v")
BIN_FIELDS = ("run", "bin_cm", "frames", "approaches", "sign_frames_pct", "line_frames_pct", "line_err_median_cm")
RUN_FIELDS = ("run", "minutes", "frames", "fps", "latency_p95_ms", "approaches", "sign_stops", "red_waits",
              "obstacle_stops", "lane_lost_s", "lane_max_cm", "battery_start_v", "battery_end_v")

_CLI_HELP = """\
One row per stop-line approach from a driving run's nav.csv (figure-eight,
course run, or any navigation run), plus per-5 cm shares by estimated
distance to the line, to paste into the workbook's TP-11 Dynamic sheet.

--sign-at / --light-at name the lap positions (1 = the first intersection of
a lap) where a sign or light stands, so sign_expected / light_expected are
filled and the sign share by distance counts only approaches with a sign.

Run from vision_stack/:
    python3 -m src.analysis.approach_log runs/routine_figure-eight_20261105_101500 --sign-at 1,8 --light-at 4,11
    make approach-log RUN=runs/routine_figure-eight_20261105_101500 ARGS="--sign-at 1,8 --light-at 4,11"
"""


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _step(label) -> int:
    head = str(label or "0").split("/")[0].strip()
    return int(head) if head.isdigit() else 0


def _p95(values):
    v = sorted(x for x in values if x is not None)
    return None if not v else round(v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], 1)


def _r(v, nd=1):
    return None if v is None else round(v, nd)


def _seconds(rows, keep) -> float:
    """Seconds spent in the frames keep() picks, each frame counting up to the next one's t."""
    total = 0.0
    for a, b in zip(rows, rows[1:] + [None]):
        if keep(a):
            t0, t1 = _num(a["t"]), _num(b["t"]) if b else None
            if t0 is not None and t1 is not None:
                total += max(t1 - t0, 0.0)
    return round(total, 2)


def reached(rows) -> list[int]:
    """Indices of the frames where the robot reached a line: the tracker leaves crossing."""
    return [i for i in range(1, len(rows)) if rows[i - 1].get("phase") == CROSSING and rows[i].get("phase") != CROSSING]


def analyze(nav_csv: Path, sign_at=(), light_at=(), lap_len: int = LAP_LEN, run: str | None = None) -> dict:
    with open(nav_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{nav_csv}: no frames")
    run = run or nav_csv.parent.name
    marks = reached(rows)
    approaches, per_frame, start = [], [], 0
    for k, r_i in enumerate(marks):
        # From the last intersection's turn (or the run's start) to this reach, driving frames only:
        # the previous stop's hold and the turn itself belong to no approach
        det = [x for x in rows[start:r_i] if x.get("rule") != RULE_INTERSECTION]
        end = marks[k + 1] if k + 1 < len(marks) else len(rows)
        turn = next((j for j in range(r_i, end) if rows[j].get("rule") == RULE_INTERSECTION), end)
        start = turn
        after = rows[r_i:max(turn, r_i + 1)]
        at = rows[r_i]
        step = max([_step(x.get("step")) for x in rows[r_i:end]] or [0])
        position = (step - 1) % lap_len + 1 if step else None
        maneuver = next((x["maneuver"] for x in rows[r_i:end] if x.get("maneuver")), "")
        left = _num(at.get("line_left_cm"))
        left = REACH_CM if left is None else left
        odo_r = _num(at.get("distance_cm"))

        def est(x):
            o = _num(x.get("distance_cm"))
            return None if o is None or odo_r is None else left + (odo_r - o)

        sign_rows = [x for x in det if _num(x.get("stop_sign"))]
        line_rows = [x for x in det if _num(x.get("stop_line_cm")) is not None]
        line_vs = [_num(x["stop_line_cm"]) - est(x) + left for x in line_rows if est(x) is not None]
        states = [x.get("drive_state", "") for x in det + after]
        worst = max(states, key=lambda s: STATE_RANK.get(s, -1)) if states else ""
        hold = _seconds(after, lambda x: x.get("reason") in HOLD_REASONS)
        red = _seconds(after, lambda x: x.get("reason") == RED_REASON)
        obst = _seconds(det, lambda x: x.get("reason") == OBSTACLE_REASON)
        stopped = "stop sign" if hold else "red light" if red else "obstacle" if obst else ""
        t0, t1 = _num(det[0]["t"]) if det else None, _num(at["t"])
        secs = None if t0 is None or t1 is None else round(t1 - t0, 2)
        lane = [abs(v) for v in (_num(x.get("lane_offset_cm")) for x in det) if v is not None]
        approaches.append({
            "run": run, "approach": k + 1, "lap": (step - 1) // lap_len + 1 if step else None, "position": position,
            "step": step or None, "maneuver": maneuver, "t_start": _r(t0, 2), "t_reach": _r(t1, 2), "seconds": secs,
            "frames": len(det), "fps": _r(len(det) / secs) if secs else None,
            "sign_expected": ("Y" if position in sign_at else "N") if sign_at else "",
            "sign_frames": len(sign_rows),
            "sign_first_est_cm": _r(est(sign_rows[0])) if sign_rows else None,
            "sign_last_est_cm": _r(est(sign_rows[-1])) if sign_rows else None,
            "line_frames": len(line_rows),
            "line_first_cm": _r(_num(line_rows[0]["stop_line_cm"])) if line_rows else None,
            "line_first_est_cm": _r(est(line_rows[0])) if line_rows else None,
            # reported - estimated, + the line left at reach: equals the true stop gap when both are right
            "line_vs_odom_cm": _r(statistics.median(line_vs), 2) if line_vs else None,
            "left_at_reach_cm": _r(left),
            "light_expected": ("Y" if position in light_at else "N") if light_at else "",
            "light_read": {"stop": "red", "caution": "yellow", "go": "green / none"}.get(worst, worst),
            "red_frames": sum(s == "stop" for s in states), "yellow_frames": sum(s == "caution" for s in states),
            "stopped_for": stopped, "hold_s": hold, "red_wait_s": red, "obstacle_s": obst,
            "lane_max_cm": _r(max(lane), 2) if lane else None,
            "lane_lost_frames": sum(x.get("lane_status") not in ("vision", "") for x in det),
            "latency_p95_ms": _p95(_num(x.get("latency_ms")) for x in det),
            "battery_v": _r(_num(at.get("battery_v")), 2),
        })
        signed = (position in sign_at) if sign_at else True
        for x in det:
            e = est(x)
            if e is not None and 0 <= e < BIN_MAX_CM:
                per_frame.append((int(e // BIN_CM), k, signed, bool(_num(x.get("stop_sign"))),
                                  _num(x.get("stop_line_cm")), e))
    bins = []
    for b in sorted({p[0] for p in per_frame}):
        fr = [p for p in per_frame if p[0] == b]
        sg = [p for p in fr if p[2]]
        errs = [p[4] - p[5] for p in fr if p[4] is not None]
        bins.append({"run": run, "bin_cm": f"{b * BIN_CM:g}-{(b + 1) * BIN_CM:g}", "frames": len(fr),
                     "approaches": len({p[1] for p in fr}),
                     "sign_frames_pct": _r(100 * sum(p[3] for p in sg) / len(sg)) if sg else None,
                     "line_frames_pct": _r(100 * sum(p[4] is not None for p in fr) / len(fr)),
                     "line_err_median_cm": _r(statistics.median(errs), 2) if errs else None})
    t = [v for v in (_num(x.get("t")) for x in rows) if v is not None]
    minutes = (t[-1] - t[0]) / 60 if len(t) > 1 else 0.0
    volts = [v for v in (_num(x.get("battery_v")) for x in rows) if v is not None]
    lane_all = [abs(v) for v in (_num(x.get("lane_offset_cm")) for x in rows) if v is not None]
    summary = {"run": run, "minutes": round(minutes, 2), "frames": len(rows),
               "fps": _r(len(rows) / (minutes * 60)) if minutes else None,
               "latency_p95_ms": _p95(_num(x.get("latency_ms")) for x in rows), "approaches": len(approaches),
               "sign_stops": sum(a["stopped_for"] == "stop sign" for a in approaches),
               "red_waits": sum(a["stopped_for"] == "red light" for a in approaches),
               "obstacle_stops": sum(a["stopped_for"] == "obstacle" for a in approaches),
               "lane_lost_s": _seconds(rows, lambda x: x.get("lane_status") not in ("vision", "")),
               "lane_max_cm": _r(max(lane_all), 2) if lane_all else None,
               "battery_start_v": _r(volts[0], 2) if volts else None,
               "battery_end_v": _r(volts[-1], 2) if volts else None}
    return {"approaches": approaches, "bins": bins, "run": summary}


def nav_files(args) -> list[Path]:
    found = []
    for a in args or [str(common.RUNS_DIR)]:
        p = Path(a)
        found += [p] if p.is_file() else sorted(p.rglob("nav.csv")) if p.is_dir() else []
    found = [f for f in found if f.is_file() and f.stat().st_size]
    if not found:
        raise FileNotFoundError(f"no nav.csv in {', '.join(args or [str(common.RUNS_DIR)])}")
    return found


def _positions(text) -> tuple[int, ...]:
    return tuple(int(s) for s in str(text or "").replace(" ", "").split(",") if s)


def _write(path: Path, fields, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.approach_log", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run folders (nav.csv found inside), or nav.csv files")
    p.add_argument("--sign-at", default="", help="lap positions with a stop sign, e.g. 1,8")
    p.add_argument("--light-at", default="", help="lap positions with a traffic light, e.g. 4,11")
    p.add_argument("--lap-len", type=int, default=LAP_LEN, help=f"intersections per lap (default {LAP_LEN})")
    p.add_argument("--out", help="output folder (default: the first argument's folder)")
    a = p.parse_args(argv)
    try:
        files = nav_files(a.runs)
        root = Path(a.runs[0]) if a.runs else Path(common.RUNS_DIR)
        root = root.parent if root.is_file() else root
        results = [analyze(f, _positions(a.sign_at), _positions(a.light_at), a.lap_len,
                           run=str(f.parent.relative_to(root)) if f.parent != root else f.parent.name)
                   for f in files]
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else root
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "approaches.csv", APPROACH_FIELDS, [r for res in results for r in res["approaches"]])
    _write(out / "dynamic_bins.csv", BIN_FIELDS, [r for res in results for r in res["bins"]])
    _write(out / "run_summary.csv", RUN_FIELDS, [res["run"] for res in results])
    for res in results:
        s = res["run"]
        print(f"{s['run']}: {s['minutes']} min, {s['approaches']} approaches, {s['sign_stops']} sign stops, "
              f"{s['red_waits']} red waits, {s['fps']} FPS, p95 {s['latency_p95_ms']} ms")
    print(f"wrote {out / 'approaches.csv'}\nwrote {out / 'dynamic_bins.csv'}\nwrote {out / 'run_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
