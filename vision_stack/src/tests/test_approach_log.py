"""
test_approach_log.py  --  src/analysis/approach_log.py

A made-up driving run: two approaches at 10 cm/s. The first has a sign (seen
from 30 cm, the line read 1.5 cm long) and holds 2 s; the second shows a red
light and waits 1 s. Each is cut at the tracker's reached frame, distances are
counted back from it by odometry, and the bins carry the shares per 5 cm.

--software  No hardware.
"""
import csv

import pytest

from src.analysis import approach_log as al

pytestmark = pytest.mark.software

FIELDS = ("t", "rule", "phase", "line_left_cm", "step", "maneuver", "lane_status", "lane_offset_cm",
          "drive_state", "stop_sign", "stop_line_cm", "reason", "latency_ms", "distance_cm", "battery_v")
DT, SPEED = 0.05, 10.0          # s per frame, cm/s
LINE_AT = 50.0                  # odometry where the first line's near edge is
REACH = 9.0


def frame(t, odo, **kw):
    row = {k: "" for k in FIELDS}
    row.update(t=round(t, 3), rule="lane_keeping", phase="idle", step="0/0", lane_status="vision",
               lane_offset_cm=0.4, drive_state="go", stop_sign=0, reason="steer", latency_ms=45.0,
               distance_cm=round(odo, 2), battery_v=7.6)
    row.update(kw)
    return row


def approach(rows, t, odo, line_at, step, sign=False, red=False):
    """Drive to REACH cm before the line, then hold or wait, then one intersection frame."""
    while line_at - odo > REACH:
        gap = line_at - odo
        kw = {}
        if sign and gap <= 30:
            kw["stop_sign"] = 1
        if 10 <= gap <= 16:
            kw.update(stop_line_cm=round(gap + 1.5, 2), phase="approach")
        elif gap < 10:
            kw.update(phase="crossing")
        if red and gap <= 25:
            kw["drive_state"] = "stop"
        # the route counts a line once its crossing starts, before the robot reaches it
        label = step if kw.get("phase") == "crossing" else step - 1
        rows.append(frame(t, odo, step=f"{label}/28", **kw))
        t, odo = t + DT, odo + SPEED * DT
    hold_reason = "stop_sign_hold" if sign else "red_light" if red else "steer"
    for _ in range(int((2.0 if sign else 1.0) / DT)):
        rows.append(frame(t, odo, step=f"{step}/28", reason=hold_reason, line_left_cm=REACH,
                          drive_state="stop" if red else "go"))
        t += DT
    rows.append(frame(t, odo, rule="intersection", step=f"{step}/28", maneuver="left"))
    return t + DT, odo


def make_run(tmp_path):
    rows, t, odo = [], 0.0, 0.0
    t, odo = approach(rows, t, odo, LINE_AT, 1, sign=True)
    t, odo = approach(rows, t, odo + 5, odo + 5 + 40, 2, red=True)
    path = tmp_path / "run_a" / "nav.csv"
    path.parent.mkdir()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_approaches_cut_at_reach_and_counted_back_by_odometry(tmp_path):
    res = al.analyze(make_run(tmp_path), sign_at=(1,), light_at=(2,))
    a, b = res["approaches"]
    assert (a["step"], a["position"], a["maneuver"]) == (1, 1, "left")
    assert (b["step"], b["position"]) == (2, 2)       # not the next line's count, already up in a's window
    assert a["sign_expected"] == "Y" and b["sign_expected"] == "N"
    assert a["stopped_for"] == "stop sign" and a["hold_s"] == pytest.approx(2.0, abs=0.06)
    assert a["sign_first_est_cm"] == pytest.approx(30.0, abs=0.6)
    # reported 1.5 cm long against the odometry: the gap at reach (9) + 1.5
    assert a["line_vs_odom_cm"] == pytest.approx(REACH + 1.5, abs=0.1)
    assert b["light_expected"] == "Y" and b["light_read"] == "red"
    assert b["stopped_for"] == "red light" and b["red_wait_s"] == pytest.approx(1.0, abs=0.06)
    assert b["sign_frames"] == 0


def test_bins_carry_bench_shares_by_distance(tmp_path):
    res = al.analyze(make_run(tmp_path), sign_at=(1,))
    by = {r["bin_cm"]: r for r in res["bins"]}
    assert by["10-15"]["line_frames_pct"] == 100.0
    assert by["10-15"]["line_err_median_cm"] == pytest.approx(1.5, abs=0.05)
    assert by["20-25"]["line_frames_pct"] == 0.0
    assert by["25-30"]["sign_frames_pct"] == 100.0       # only the signed approach counts
    assert by["35-40"]["sign_frames_pct"] == 0.0
    # 5 cm at 0.5 cm a frame: the first stop's hold isn't counted into the second approach
    assert (by["45-50"]["frames"], by["45-50"]["approaches"]) == (10, 1)
    s = res["run"]
    assert (s["approaches"], s["sign_stops"], s["red_waits"]) == (2, 1, 1)


def test_cli_writes_the_three_files(tmp_path):
    make_run(tmp_path)
    assert al.main([str(tmp_path), "--sign-at", "1", "--out", str(tmp_path / "out")]) == 0
    for name in ("approaches.csv", "dynamic_bins.csv", "run_summary.csv"):
        assert (tmp_path / "out" / name).stat().st_size
    assert al.main([str(tmp_path / "empty")]) == 1
