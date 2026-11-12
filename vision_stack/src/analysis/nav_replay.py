"""Navigation replay: today's Navigation over a recorded run's packets, against what the robot did.

Purpose:
    A rule change (a sign threshold, a light memory) is cheap to test on a
    run already driven: p3.csv logs every EstimationPacket field per frame,
    so the robot's current Navigation can be fed the same packets and its
    stops set beside the ones the robot made (nav.csv). It is open loop:
    the packets are what the robot saw driving its own way, so a rule that
    would have driven differently is judged on frames it might not have
    seen. Two parts can't be replayed that way and are off: the wheel stall
    guard (the wheels answer the recorded commands, not the replay's) and
    the end of course (without the recorded robot's holds, the same lost
    lane counts toward the end and stops every rule after it).

Main package:
    replay(p3, route, ...) -> per-frame (frame_id, rule, reason, step);
    compare(recorded, replayed) -> each stop onset kept, added or removed;
    nav_replay.json beside nav.csv.
"""
import argparse
import csv
import json
import sys
from dataclasses import MISSING, fields
from pathlib import Path

from src.analysis import common
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation import RULE_OBSTACLE, RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT, Navigation
from src.navigation.route import MANEUVERS, Route

# The rules whose onsets are compared: each is a stop the robot makes
STOP_RULES = (RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT, RULE_OBSTACLE)
# packet field -> p3.csv column, where they differ
P3_COLUMNS = {"lane_mode": "p2_mode"}
TEXT_FIELDS = {"lane_status": "", "drive_state": "go", "lane_mode": "none"}

_CLI_HELP = """\
Replay today's Navigation over a recorded run's packets (p3.csv) and set its
stops beside the robot's (nav.csv): each stop-sign, red-light and obstacle stop
kept, added or removed, with its route step. Open loop, with the wheel stall
guard and the end of course off (see the module header). --sign-min-frames
tries another stop-sign threshold for this replay only.

Run from vision_stack/:
    python3 -m src.analysis.nav_replay runs/routine_figure-eight_20261104_165132/attempt_02
    make nav-replay RUN=runs/routine_figure-eight_20261104_165132/attempt_02 ARGS="--sign-min-frames 1"
"""


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def packets(p3_path):
    """One EstimationPacket per p3.csv row; a column the log lacks takes the field's default."""
    defaults = {f.name: f.default for f in fields(EstimationPacket)}
    types = {f.name: str(f.type) for f in fields(EstimationPacket)}
    with open(p3_path, newline="") as fh:
        for row in csv.DictReader(fh):
            kw = {}
            for name, t in types.items():
                v = row.get(P3_COLUMNS.get(name, name), "")
                if name in ("frame_id", "timestamp_ms"):
                    kw[name] = int(float(v))
                elif "bool" in t:
                    kw[name] = v in ("True", "1", "true")
                elif name in TEXT_FIELDS:
                    kw[name] = v or TEXT_FIELDS[name]
                elif "None" in t:
                    kw[name] = _num(v)
                else:
                    x = _num(v)
                    kw[name] = x if x is not None else (0.0 if defaults[name] is MISSING else defaults[name])
            yield EstimationPacket(**kw)


def route_of(nav_rows) -> Route:
    """The route as driven: each step's maneuver, from nav.csv's step and maneuver columns."""
    seen, maneuvers = set(), []
    for r in nav_rows:
        step = r.get("step", "")
        if step and " " in step and step not in seen:
            seen.add(step)
            m = r.get("maneuver") or step.split()[-1]
            maneuvers.append(m if m in MANEUVERS else "straight")
    return Route(tuple(maneuvers))


def replay(p3_path, route: Route, gyro_bias_dps: float, sign_min_frames: int | None = None) -> list[tuple]:
    """(frame_id, rule, reason, step) per frame, from today's Navigation, open loop."""
    nav = Navigation(gyro_bias_dps=gyro_bias_dps, route=route)
    if sign_min_frames is not None:
        nav.rules[1][1].sign_min_frames = max(1, sign_min_frames)
    nav.stall.observe = lambda *a, **k: False          # the wheels answer the recorded commands
    nav._end.end_stale_ms = float("inf")               # a lost lane can't end an open-loop replay
    out = []
    for p in packets(p3_path):
        nav.update(p)
        rec = nav.record
        out.append((p.frame_id, rec.get("rule"), rec.get("reason"), rec.get("step") or ""))
    return out


def onsets(frames) -> list[dict]:
    """Each stop: the frame a STOP_RULES rule takes over, with its step number."""
    out, prev = [], None
    for fid, rule, reason, step in frames:
        if rule in STOP_RULES and rule != prev:
            out.append({"frame": fid, "rule": rule, "step": (step or "").split("/")[0]})
        prev = rule
    return out


def compare(recorded: list[dict], replayed: list[dict]) -> list[dict]:
    """Stops matched by (rule, step): kept in both, removed (recorded only) or added (replay only)."""
    key = lambda s: (s["rule"], s["step"])
    rec, rep = {key(s): s for s in recorded}, {key(s): s for s in replayed}
    rows = []
    for k in sorted(set(rec) | set(rep), key=lambda k: (int(k[1]) if k[1].isdigit() else 10 ** 6, k[0])):
        status = "kept" if k in rec and k in rep else ("removed" if k in rec else "added")
        rows.append({"rule": k[0], "step": k[1], "status": status,
                     "recorded_frame": rec[k]["frame"] if k in rec else None,
                     "replay_frame": rep[k]["frame"] if k in rep else None})
    return rows


def analyze(folder, gyro_bias_dps: float, sign_min_frames: int | None = None) -> dict:
    folder = Path(folder)
    nav_csv, p3_csv = common.find_csv(folder, "nav.csv"), common.find_csv(folder, "p3.csv")
    with open(nav_csv, newline="") as fh:
        nav_rows = list(csv.DictReader(fh))
    recorded = onsets((int(r["frame_id"]), r.get("rule"), r.get("reason"), r.get("step", "")) for r in nav_rows)
    replayed = onsets(replay(p3_csv, route_of(nav_rows), gyro_bias_dps, sign_min_frames))
    rows = compare(recorded, replayed)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("kept", "removed", "added")}
    return {"nav_csv": str(nav_csv), "p3_csv": str(p3_csv), "sign_min_frames": sign_min_frames,
            "off": ["wheel stall guard", "end of course"], "stops": rows, "counts": counts}


def report_lines(res: dict) -> list[str]:
    c = res["counts"]
    lines = [f"replayed {res['p3_csv']}",
             f"open loop; off: {', '.join(res['off'])}"
             + (f"; stop-sign minimum {res['sign_min_frames']}" if res["sign_min_frames"] else ""),
             f"stops: {c['kept']} kept, {c['removed']} removed, {c['added']} added",
             f"  {'step':>5} {'rule':<14} {'status':<8} {'robot':>7} {'replay':>7}"]
    for r in res["stops"]:
        frame = lambda f: "--" if f is None else str(f)
        lines.append(f"  {r['step']:>5} {r['rule']:<14} {r['status']:<8} "
                     f"{frame(r['recorded_frame']):>7} {frame(r['replay_frame']):>7}")
    return lines


def main(argv=None) -> int:
    from src.config import GYRO_BIAS_DPS
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.nav_replay", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="a run or attempt folder with nav.csv and p3.csv (default: the newest)")
    p.add_argument("--sign-min-frames", type=int, default=None,
                   help="the stop-sign rule's votes a sighting, for this replay only (1: the rule before 2026-11-12)")
    p.add_argument("--gyro-bias", type=float, default=GYRO_BIAS_DPS, help="gyro Z at rest, deg/s")
    a = p.parse_args(argv)
    try:
        res = analyze(a.run or common.find_csv(None, "p3.csv").parent, a.gyro_bias, a.sign_min_frames)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("\n".join(report_lines(res)))
    out = Path(res["nav_csv"]).parent / "nav_replay.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
