"""
run_fixtures.py  --  made-up run folders for the analysis tools' tests

Each writes what the real tool writes, with the real field lists: a navigation
run (nav.csv, nav_video.csv, summary.txt), a diagnostics recording (threads,
cores, system), a figure-eight attempt (through the routine's own EightWatch),
an obstacle-check run (trials.csv, results.json, saved looks), a capture-
anatomy latency.csv, a sched-latency folder and a run whose right motor dies. Numbers are chosen so each
tool's findings have something to find.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.debugger.debug_phase3 import Phase3View
from src.diagnostics.system_monitor import FIELDS as SYSTEM_FIELDS
from src.diagnostics.threads import CORE_FIELDS, THREAD_FIELDS
from src.navigation_linker import NAV_FIELDS
from src.routines import figure_eight as fe
from src.routines.obstacle_check import ObstacleCheck
from src.tests.scenes import FRAME_H, scene, with_obstacle

SUMMARY = """navigation run

[TIMING] per stage, ms:
 preprocess             med    6.1  p95    7.0
 geometry               med   20.6  p95   24.0
   geometry.edges       med    5.0  p95    6.0
 color                  med    5.2  p95    6.0
 obstacle               med    5.9  p95    7.1
 TOTAL (median)              37.8  ->  26.5 FPS

[PHASE 3]
"""


def _write(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows([{**{k: "" for k in fields}, **r} for r in rows])
    return path


def nav_run(folder, n=400, video=True, hold=(200, 300), seed=3) -> Path:
    """
    A run at ~16.5 FPS: Phase 2 ~50 ms on two_boundary frames, ~43 on left_only; a light green, yellow from
    hold[0] - 20, red through hold[1] with the light rule braking over hold; every 9th frame finds no light;
    frames 50 and 120 read a lone red.
    """
    folder = Path(folder)
    rng = np.random.default_rng(seed)
    nav, vid, buf, state, t = [], [], [], "go", 0.0
    for i in range(n):
        mode = "two_boundary" if i % 10 < 8 else "left_only"
        p2 = 1300.0 if i == 0 else (50.0 if mode == "two_boundary" else 43.0) + float(rng.normal(0, 1))
        cap, p3, nv, out = 2.0, 0.6, 0.3, 6.0
        true = "go" if i < hold[0] - 20 or i >= hold[1] else ("caution" if i < hold[0] else "stop")
        seen = i % 9 != 4
        raw = ("stop" if i in (50, 120) else true) if seen else "go"
        buf = (buf + [raw])[-3:]
        for s in ("stop", "caution", "go"):
            if buf.count(s) >= 2:
                state = s
        rule = "traffic_light" if hold[0] <= i < hold[1] else "lane_keeping"
        nav.append({"frame_id": i, "t": round(t, 4), "capture_ms": cap, "phase2_ms": round(p2, 2), "phase3_ms": p3,
                    "nav_ms": nv, "latency_ms": round(p2 + 3, 2), "rule": rule, "lane_mode": mode, "step": "1/3",
                    "drive_state": state, "lane_status": "vision", "reason": "red_light" if rule == "traffic_light" else "",
                    "gc_ms": 12.0 if i % 100 == 7 else ""})
        label = {"stop": "red", "caution": "yellow", "go": "green"}[raw]
        vid.append({"frame_id": i, "traffic_detections": f"{label}:0.900:pass" if seen else "",
                    "traffic_raw": raw, "drive_state": state})
        t += (cap + p2 + p3 + nv + out) / 1000.0
    _write(folder / "nav.csv", NAV_FIELDS, nav)
    if video:
        _write(folder / "nav_video.csv", Phase3View.CSV_FIELDS, vid)
    (folder / "summary.txt").write_text(SUMMARY)
    return folder


def diag(folder, task0=35.0, main=85.0, n=120, seed=1) -> Path:
    """A recording: idle for 5 s, then main busy until 5 s before the end; task0 appears at 3 s."""
    folder = Path(folder)
    rng = np.random.default_rng(seed)
    th, co, sy = [], [], []
    for i in range(n):
        t, busy = 0.5 * i, 10 <= i < n - 10
        for tid, name, base in ((1, "main", main if busy else 5.0), (2, "task0", task0), (3, "python", 40.0 if busy else 0.0),
                                (4, "pigpio-cb", 6.0)):
            if name == "task0" and i < 6:
                continue
            th.append({"elapsed_s": t, "tid": tid, "name": name, "core": tid % 4,
                       "cpu_pct": round(max(0.0, base + float(rng.normal(0, 2))), 2)})
        for c in range(4):
            co.append({"elapsed_s": t, "core": c, "busy_pct": round((main + task0 + 46) / 4, 1)})
        sy.append({"elapsed_s": t, "temp_c": 47.0, "cpu_mhz": 1000, "rss_mb": 180, "mem_available_mb": 210})
    _write(folder / "threads.csv", THREAD_FIELDS, th)
    _write(folder / "cores.csv", CORE_FIELDS, co)
    _write(folder / "system.csv", SYSTEM_FIELDS, sy)
    return folder


def figure_eight(folder, laps=3, slow_corner=3, seed=5, touch_frame=None) -> Path:
    """
    An attempt: lefts at ~28 deg/s, rights at ~42; at slow_corner the chassis sits still for its first
    30 frames with the wheels driving, so those turns end on the time limit. Written by EightWatch itself.
    """
    folder = Path(folder)
    rng = np.random.default_rng(seed)
    rows, t, step = [], 0.0, 0

    def frame(**kw):
        nonlocal t
        rows.append({"t": round(t, 3), "step": f"{step}/400", **kw})
        t += 0.06
    for step in range(1, laps * fe.ROUTE_LAP.__len__() + 1):
        m = fe.ROUTE_LAP[(step - 1) % len(fe.ROUTE_LAP)]
        sign = -1 if m == "left" else 1
        for _ in range(50):
            frame(rule="lane_keeping", lane_status="vision", left_cps=1100, right_cps=1100, yaw_rate=0, brake=0)
        stalled = (step - 1) % len(fe.ROUTE_LAP) + 1 == slow_corner
        limit, rate = (4.1, 28.0) if m == "left" else (2.4, 42.0)
        h, k = 0.0, 0
        while abs(h) < 85 and k * 0.06 < limit:
            yaw = rate * (0.1 if stalled and k < 30 else 1.0) + float(rng.normal(0, 0.5))
            h += sign * yaw * 0.06
            frame(rule="intersection", stage="turn", maneuver=m, heading_deg=round(h, 1),
                  left_cps=600 if m == "left" else 1700, right_cps=1700 if m == "left" else 600,
                  yaw_rate=round(sign * yaw, 1), brake=0)
            k += 1
        end = "gyro target" if abs(h) >= 85 else "time limit"
        for _ in range(10):
            frame(rule="intersection", stage="exit", maneuver=m, turn_end=end, turn_deg=round(h, 1), heading_deg=0.0,
                  left_cps=1100, right_cps=1100, yaw_rate=0, brake=0)
    presses = iter([0] * (touch_frame or 0) + [1] + [0] * len(rows)) if touch_frame is not None else None
    watch = fe.EightWatch(seconds=1e9, touches=(lambda: next(presses)) if presses else None)
    for r in rows:
        watch(r)
    _write(folder / "nav.csv", NAV_FIELDS, rows)
    fe.FigureEight._write(folder, watch, watch.slips())
    return folder


def obstacle_run(root, name, heights, seen, color=(40, 40, 210), gaps=(10, 15, 20, 25, 30), alarms=0) -> Path:
    """One object's run: a box of the given height (px; 0 = none) at each gap, the saved looks beside."""
    folder = Path(root) / f"routine_obstacle-check_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for g, h, s in zip(gaps, heights, seen):
        base = FRAME_H - int((g - 10) * 3)
        frame = with_obstacle(scene(), 200, 290, base - h, base, bgr=color) if h else scene()
        cv2.imwrite(str(folder / f"attempt_01_{g}cm.jpg"), frame)
        rows.append({"gap_cm": g, "lateral_cm": 0, "state": "present", "frames": 40, "seen_pct": 90 if s else 0,
                     "voted": s, "reported_cm": g + 0.8 if s else "", "error_cm": 0.8 if s else "", "clipped_pct": 0,
                     "would_brake": s and g <= 12, "result": "ok" if s else "missed"})
    for k in range(3):
        rows.append({"state": "clear", "frames": 40, "seen_pct": 0, "voted": k < alarms, "clipped_pct": 0,
                     "would_brake": False, "result": "wrong" if k < alarms else "ok"})
    _write(folder / "trials.csv", ObstacleCheck.fields, rows)
    (folder / "results.json").write_text(json.dumps({"options": {"object": name}, "criteria": [], "rows": rows}))
    return folder


def latency(folder, convert=(14.0, 18.0), flip_period=True, n=200, seed=1) -> Path:
    """capture_anatomy's latency.csv: camera ~11 ms; videoconvert stepping at n/2; videoflip at the frame period."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ts = 1000 + 50 * i
        rows += [{"ts_ms": ts, "kind": "camera", "element": "libcamerasrc0", "ms": round(11 + float(rng.normal(0, 0.3)), 3)},
                 {"ts_ms": ts + 12, "kind": "element", "element": "videoconvert0",
                  "ms": round(convert[i >= n // 2] + float(rng.normal(0, 0.3)), 3)},
                 {"ts_ms": ts + 30, "kind": "element", "element": "videoflip0",
                  "ms": round((50.0 if flip_period else 2.0) + float(rng.normal(0, 0.2)), 3)},
                 {"ts_ms": ts + 31, "kind": "pipeline", "element": "pipeline0", "ms": round(22 + float(rng.normal(0, 0.5)), 3)}]
    return _write(Path(folder) / "latency.csv", ("ts_ms", "kind", "element", "ms"), rows)


def sched(folder, label="", worst=(5332, 3999, 139), seed=2) -> Path:
    """A sched-latency folder: sched_latency.json and histogram.csv, each case's worst as given (us)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    spread = {"python": rng.gamma(4, 25, 3000), "other": rng.gamma(3, 25, 3000), "fifo": rng.gamma(2, 8, 3000)}
    data = {c: np.r_[np.minimum(v, w - 1), w] for (c, v), w in zip(spread.items(), worst)}
    hist = {c: Counter(np.round(v).astype(int).tolist()) for c, v in data.items()}
    us = sorted(set().union(*[h.keys() for h in hist.values()]))
    _write(folder / "histogram.csv", ("us_late", "python", "other", "fifo"),
           [{"us_late": u, **{c: hist[c].get(u, 0) for c in hist}} for u in us])
    cases = {c: {"n": len(v), "min": int(v.min()), "p50": int(np.percentile(v, 50)), "p99": int(np.percentile(v, 99)),
                 "p999": int(np.percentile(v, 99.9)), "max": int(v.max())} for c, v in data.items()}
    (folder / "sched_latency.json").write_text(json.dumps({"label": label, "seconds": 3, "cases": cases, "skipped": {}}))
    return folder


def dying_motor(folder, weak_at=60.0, dead_at=100.0, n=2400, seed=4) -> Path:
    """
    A 120 s run at 20 FPS, both wheels at 3300 counts/s per duty, braked for 1 s every 20 s and reading 0 on the
    frame after (a start from rest); from weak_at the right wheel turns 25% slower and from dead_at not at all,
    its duty pushed to 1.0 as the PI would. Each frame's speed answers the previous frame's command.
    """
    rng = np.random.default_rng(seed)
    rows, before = [], (0.0, 0.0, 1)
    for i in range(n):
        t = 0.05 * i
        brake = int(t % 20 < 1.0)
        duty = 0.0 if brake else 0.4 + 0.25 * (t % 10 > 6)
        right = 1.0 if t >= dead_at and not brake else duty
        share = 0.0 if t >= dead_at else 0.75 if t >= weak_at else 1.0
        rest = before[2] or (rows and rows[-1]["brake"])        # the first frame after a brake
        cps = (0.0, 0.0) if rest else (3300 * before[0] + float(rng.normal(0, 20)), share * 3300 * before[1])
        rows.append({"frame_id": i, "t": round(t, 3), "rule": "intersection" if brake else "lane_keeping",
                     "brake": brake, "cmd_left": duty, "cmd_right": right,
                     "left_cps": round(cps[0], 1), "right_cps": round(cps[1], 1)})
        before = (duty, right, brake)
    return _write(Path(folder) / "nav.csv", NAV_FIELDS, rows)
