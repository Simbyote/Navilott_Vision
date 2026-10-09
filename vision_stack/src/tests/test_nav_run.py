"""
test_nav_run.py  --  src/analysis/nav_run.py

On hand-built nav.csv rows with known answers: time and episodes per rule
and the transitions; braking and its reasons; lane keeping's lane states,
offset in cm or normalized, weaving counted only inside lane keeping and
past the deadband, steering at its limit; an intersection grouped across
the stop sign's hold, its stage times, turn end, the rule's turn heading
and the gyro's net of the bias, and the window after it (offset, a veer
against the run's p95, both lines back); wheel balance at equal duty only;
late frames; every finding on its own. End to end on intersection_linker's
synthetic left turn, and the command line.

--software  CSV files in a temp folder. No camera or robot.
"""
import csv
import json

import pytest

import src.analysis.nav_run as nr
from src.navigation.intersection import ADVANCE_MS
from src.analysis.common import Table
from src.navigation.route import LEFT
from src.navigation_linker import NAV_FIELDS

DT = 0.05


def row(i, **kw):
    """One nav.csv row: lane keeping on vision, straight, both wheels at 300 cps, unless kw says otherwise."""
    r = {f: "" for f in NAV_FIELDS}
    r.update(frame_id=i, t=round(i * DT, 3), rule="lane_keeping", lane_status="vision", lane_mode="two_boundary",
             lane_offset=0.0, steer=0.0, cmd_left=0.4, cmd_right=0.4, brake=0, left_cps=300.0, right_cps=300.0,
             yaw_rate=0.0, reason="steer", latency_ms=5.0)
    r.update(kw)
    return r


def table(tmp_path, rows):
    path = tmp_path / "nav.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, NAV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return Table(path)


def lane(n, start=0, **kw):
    return [row(start + i, **kw) for i in range(n)]


# =============================================================================
# Rules, braking, lane keeping
# =============================================================================

@pytest.mark.software
def test_time_and_episodes_per_rule_and_the_transitions(tmp_path):
    rows = lane(10) + lane(4, 10, rule="stop_sign", brake=1, reason="stop_sign_hold") + lane(6, 14)
    res = nr.analyze(table(tmp_path, rows))
    by = res["rules"]["by_rule"]
    assert by["lane_keeping"] == {"seconds": 0.8, "share": 0.8, "episodes": 2}
    assert by["stop_sign"] == {"seconds": 0.2, "share": 0.2, "episodes": 1}
    assert res["rules"]["transitions"] == {"lane_keeping -> stop_sign": 1, "stop_sign -> lane_keeping": 1}
    assert res["braking"] == {"seconds": 0.2, "share": 0.2, "episodes": 1, "reasons": {"stop_sign_hold": 4}}


@pytest.mark.software
def test_lane_keeping_offset_weaving_and_steering_at_its_limit(tmp_path):
    steer = [0.1, -0.1, 0.1, 0.01, -0.01, 0.4, 0.4, -0.1]          # 0.01 / -0.01 are inside the deadband
    rows = [row(i, steer=s, lane_offset=o) for i, (s, o) in enumerate(zip(steer, [0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1]))]
    rows += [row(8, lane_status="hold"), row(9, lane_status="stale")]
    lk = nr.analyze(table(tmp_path, rows))["lane_keeping"]
    assert lk["offset_unit"] == "norm" and lk["offset"]["n"] == 8
    assert lk["offset"]["mean"] == pytest.approx(0.125) and lk["offset"]["max_abs"] == pytest.approx(0.3)
    # signs past the deadband: + - + + + - : 3 changes in 0.5 s of lane keeping
    assert lk["weave_per_s"] == 6.0
    assert lk["saturated_share"] == 0.2 and lk["lane_status"] == {"vision": 0.8, "hold": 0.1, "stale": 0.1}


@pytest.mark.software
def test_the_offset_is_in_cm_when_the_run_has_a_scale(tmp_path):
    rows = lane(4, lane_offset=0.1, lane_offset_cm=1.5)
    lk = nr.analyze(table(tmp_path, rows))["lane_keeping"]
    assert lk["offset_unit"] == "cm" and lk["offset"]["mean"] == 1.5


@pytest.mark.software
def test_weaving_is_never_counted_across_a_break_in_lane_keeping(tmp_path):
    rows = lane(3, steer=0.1) + [row(3, rule="intersection", steer=-0.1), row(4, rule="intersection", steer=0.1)] \
        + lane(3, 5, steer=0.1)
    assert nr.analyze(table(tmp_path, rows))["lane_keeping"]["weave_per_s"] == 0.0


# =============================================================================
# Intersections
# =============================================================================

def crossing(start, bias_yaw=0.0):
    """to_line 10 frames, held by a stop sign 6, turn 20 at -60 deg/s, exit 4: one intersection."""
    rows = [row(start + i, rule="intersection", stage="to_line", step="1/1 left", maneuver="left",
                lane_status="stale", yaw_rate=bias_yaw) for i in range(10)]
    rows += [row(start + 10 + i, rule="stop_sign", brake=1, reason="stop_sign_hold", step="1/1 left",
                 yaw_rate=bias_yaw) for i in range(6)]
    rows += [row(start + 16 + i, rule="intersection", stage="turn", step="1/1 left", maneuver="left",
                 heading_deg=-3.0 * (i + 1), yaw_rate=-60.0 + bias_yaw, cmd_left=0.36, cmd_right=0.63)
             for i in range(20)]
    rows += [row(start + 36 + i, rule="intersection", stage="exit", step="1/1 left", maneuver="left",
                 turn_end="gyro target", heading_deg=0.0, yaw_rate=bias_yaw) for i in range(4)]
    return rows


@pytest.mark.software
def test_an_intersection_is_grouped_across_the_stop_and_measured(tmp_path):
    rows = lane(10, lane_offset=0.02) + crossing(10, bias_yaw=1.0) + lane(10, 50, lane_offset=0.02,
                                                                          lane_mode="left_only") + lane(30, 60)
    (x,) = nr.analyze(table(tmp_path, rows), gyro_bias_dps=1.0)["intersections"]
    assert x["step"] == "1/1 left" and x["maneuver"] == "left" and x["start_s"] == 0.5 and x["seconds"] == 2.0
    assert x["stage_s"] == {"to_line": 0.5, "advance": 0.0, "turn": 1.0, "exit": 0.2} and x["held_s"] == 0.3
    assert x["turn_end"] == "gyro target" and x["turn_deg"] == -60.0
    assert x["turned_deg"] == -60.0                                     # 1 s at -60, net of the 1 deg/s bias
    assert x["after"]["lane_back_s"] == 0.55                            # left_only for 10 frames first
    assert x["after"]["seconds"] == 2.0 and not x["after"]["veer"]


@pytest.mark.software
def test_a_veer_after_the_crossing_is_flagged_against_the_runs_p95(tmp_path):
    # held frames repeating a stale offset aren't normal lane keeping: they mustn't raise the baseline
    rows = lane(35, lane_offset=0.02) + lane(5, 35, lane_status="hold", lane_offset=0.5) + crossing(40) \
        + [row(80 + i, lane_offset=0.02 + 0.03 * i) for i in range(10)]
    (x,) = nr.analyze(table(tmp_path, rows))["intersections"]
    assert x["after"]["veer"] and x["after"]["offset_max_abs"] == pytest.approx(0.29)


# =============================================================================
# Wheels, latency, findings
# =============================================================================

@pytest.mark.software
def test_wheel_balance_only_counts_equal_duty_driving_frames(tmp_path):
    rows = lane(10, left_cps=330.0, right_cps=300.0) + lane(5, 10, cmd_left=0.3, cmd_right=0.5, left_cps=100.0) \
        + lane(5, 15, brake=1, left_cps=0.0, right_cps=0.0)
    wb = nr.analyze(table(tmp_path, rows))["wheel_balance"]
    assert wb["frames"] == 10 and wb["imbalance_pct"] == pytest.approx(9.5, abs=0.1)


@pytest.mark.software
def test_frames_later_than_one_and_a_half_budgets_are_counted(tmp_path):
    rows = lane(5)
    rows[3]["t"] = 0.2                      # 0.1 s after the frame before it: over 1.5 budgets (75 ms)
    rows[4]["t"] = 0.25
    assert nr.analyze(table(tmp_path, rows))["latency"]["late_frames"] == 1


@pytest.mark.software
@pytest.mark.parametrize("rows, words", [
    (lambda: lane(10, lane_offset=0.2), "right of center"),
    (lambda: [row(i, steer=0.1 if i % 2 else -0.1) for i in range(20)], "weaves"),
    (lambda: lane(8) + lane(2, 8, lane_status="stale"), "stale"),
    (lambda: lane(8) + lane(2, 8, steer=0.4), "at its limit"),
    (lambda: lane(10, left_cps=200.0, right_cps=300.0), "right wheel turns"),
    (lambda: lane(10, latency_ms=80.0), "latency"),
    (lambda: lane(5) + lane(1, 5, reason="contract", brake=1), "broke the contract"),
    (lambda: lane(40, lane_offset=0.02) + crossing(40) + [row(80 + i, lane_offset=0.3) for i in range(10)], "veered"),
    (lambda: lane(10) + [dict(r, turn_end="time limit") if r["turn_end"] else r for r in crossing(10)] + lane(10, 50),
     "time limit"),
    (lambda: lane(10) + [dict(r, yaw_rate=-20.0) if r["stage"] == "turn" else r for r in crossing(10)] + lane(10, 50),
     "expected -90"),
])
def test_each_finding_on_its_own(tmp_path, rows, words):
    found = nr.analyze(table(tmp_path, rows()), gyro_bias_dps=0.0)["findings"]
    assert any(words in f for f in found), found


@pytest.mark.software
def test_a_clean_run_has_no_findings(tmp_path):
    assert nr.analyze(table(tmp_path, lane(40)))["findings"] == []


# =============================================================================
# End to end and the command line
# =============================================================================

@pytest.mark.software
def test_intersection_linkers_left_turn_reads_as_one_clean_gyro_turn(tmp_path):
    from src.tests.test_intersection_linker import go
    go(tmp_path, LEFT)
    res = nr.analyze(Table(tmp_path / LEFT / "nav.csv"), gyro_bias_dps=0.0)
    (x,) = res["intersections"]
    assert x["maneuver"] == "left" and x["turn_end"] == "gyro target" and x["stage_s"]["turn"] > 1.0
    assert x["stage_s"]["advance"] == pytest.approx(ADVANCE_MS / 1000, abs=0.1)        # on into it first
    assert x["turned_deg"] == pytest.approx(-90, abs=10) and x["after"]["lane_back_s"] is not None
    assert res["findings"] == []


@pytest.mark.software
def test_the_command_line_writes_its_summary_next_to_the_run(tmp_path, capsys):
    table(tmp_path, lane(10) + crossing(10) + lane(10, 50))
    assert nr.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "intersection  1/1 left" in out and "findings" in out
    assert json.loads((tmp_path / "nav_run.json").read_text())["intersections"][0]["step"] == "1/1 left"
    (tmp_path / "p3.csv").write_text("frame_id,lane_offset\n0,0.1\n")
    assert nr.main([str(tmp_path / "p3.csv")]) == 1                     # not a nav.csv
