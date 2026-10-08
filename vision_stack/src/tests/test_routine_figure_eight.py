"""
test_routine_figure_eight.py  --  src/routines/figure_eight.py

EightWatch over generated laps of nav.csv rows: each intersection's
maneuver from the lap's order, its turn end, heading and offset, the lane
coming back, lane-lost time, contract brakes, lap times and volts, ending
at the time; the routine on a fake rig, battery and run: one continuous run
with the repeated route and the battery, the tester's touches and laps,
intersections.csv and laps.csv per attempt, a run ended mid-turn not
counting that lap, the rig and battery closed at the Enter prompt; the
criteria one at a time.

--software  A scripted tester, a fake rig, battery and navigation run.
"""
import csv
import json

import pytest

import src.routines.figure_eight as fe
import src.routines.harness as h
from src.intersection_linker import HEADING_TOLERANCE_DEG
from src.navigation.intersection import TURN_END_GYRO, TURN_END_TIME
from src.navigation.route import RIGHT
from src.tests.test_routines import Person, no_conditions

DT = 0.05


def lap_rows(laps=1, t0=0.0, step0=0, heading=lambda step, m: fe.EXPECTED_DEG[m], turn_end=TURN_END_GYRO,
             volts=lambda step: 12.0 - step * 0.01, lane_back=True, lost_frames=0):
    """Rows for laps of the figure 8: per intersection 10 frames of the turn, then lane keeping."""
    rows, t, total = [], t0, len(fe.ROUTE_LAP) * laps
    for k in range(total):
        step = step0 + k + 1
        m = fe.ROUTE_LAP[(step - 1) % 8]
        label = f"{step}/1600 {m}"
        for f in range(10):
            rows.append({"t": round(t, 2), "rule": "intersection", "step": label, "stage": "turn",
                         "turn_end": turn_end if f >= 8 else "", "heading_deg": heading(step, m) * (f + 1) / 10,
                         "lane_status": "stale", "reason": "turn", "battery_v": volts(step)})
            t += DT
        for f in range(lost_frames):
            rows.append({"t": round(t, 2), "rule": "lane_keeping", "step": label, "lane_status": "hold",
                         "reason": "steer", "battery_v": volts(step)})
            t += DT
        for f in range(10 if lane_back else 0):
            rows.append({"t": round(t, 2), "rule": "lane_keeping", "step": label, "lane_status": "vision",
                         "reason": "steer", "battery_v": volts(step)})
            t += DT
    return rows


def feed(watch, rows):
    for r in rows:
        why = watch(r)
        if why:
            return why
    return None


# =============================================================================
# EightWatch
# =============================================================================

@pytest.mark.software
def test_each_intersection_its_maneuver_turn_heading_and_lane():
    w = fe.EightWatch(seconds=1e9)
    assert feed(w, lap_rows(laps=2)) is None
    assert [i["maneuver"] for i in w.intersections] == list(fe.ROUTE_LAP) * 2
    first, fifth = w.intersections[0], w.intersections[4]
    assert (first["step"], first["lap"], first["turn_end"], first["heading_deg"], first["heading_off_deg"]) == \
        (1, 1, TURN_END_GYRO, -90.0, 0.0)
    assert (fifth["maneuver"], fifth["heading_deg"], fifth["lane_back"], fifth["battery_v"]) == (RIGHT, 90.0, True, 11.95)
    assert w.intersections[8]["lap"] == 2 and len(w.finished()) == 16
    assert w.frames == 16 * 20 and w.lane_lost_s == 0.0 and w.contract == 0


@pytest.mark.software
def test_laps_run_from_one_lap_start_to_the_next():
    w = fe.EightWatch(seconds=1e9)
    feed(w, lap_rows(laps=3))
    assert w.laps() == [{"lap": 1, "t_start": 0.0, "seconds": 8.0, "battery_v": 11.99},
                        {"lap": 2, "t_start": 8.0, "seconds": 8.0, "battery_v": 11.91}]   # the 3rd has no next start


@pytest.mark.software
def test_lane_lost_time_contract_brakes_and_the_time_limit():
    w = fe.EightWatch(seconds=1e9)
    rows = lap_rows(laps=1, lost_frames=4)
    rows.insert(5, {"t": rows[5]["t"], "rule": "lane_keeping", "step": rows[5]["step"], "lane_status": "vision",
                    "reason": "contract"})
    feed(w, rows)
    assert w.lane_lost_s == pytest.approx(8 * 4 * DT) and w.contract == 1
    timed = fe.EightWatch(seconds=3.0)
    assert feed(timed, lap_rows(laps=1)) == fe.TIME_UP and timed.t == 3.0


@pytest.mark.software
def test_rows_before_the_first_intersection_and_odd_labels_are_harmless():
    w = fe.EightWatch(seconds=1e9)
    assert w({"t": 0.0, "rule": "lane_keeping", "step": "0/1600", "lane_status": "hold"}) is None
    assert w({"t": 0.05, "rule": "lane_keeping", "step": None, "lane_status": "vision"}) is None
    assert w.intersections == [] and fe._step("x/1") == 0 and fe._step("12/1600 left") == 12


# =============================================================================
# The routine
# =============================================================================

class Rig:
    def __init__(self, log):
        self.log = log

    def __call__(self, camera, motors, button):
        self.log.append(("open", camera, motors, button))
        log = self.log
        return (type("S", (), {"close": lambda s: log.append("source closed")})(),
                type("Se", (), {"stop": lambda s: log.append("sensors stopped")})(),
                type("M", (), {"stop": lambda s: log.append("motors stopped")})(), None)


class Battery:
    def __init__(self, log):
        self.log = log

    def cleanup(self):
        self.log.append("battery released")


def fake_run(rows, log, wall=None):
    def run(source, sensors, motor, nav, config, p3, out_dir, system, max_run_s, motors_on, render, stop_when,
            battery):
        log.append(("run", out_dir.rsplit("/", 1)[-1], max_run_s, len(nav.progress.route.maneuvers),
                    nav.progress.route.maneuvers[:8], motors_on, render, battery is not None))
        why = feed(stop_when, rows)
        return {"ended_by": why or "ended early (lane lost before the route was done)",
                "run": {"wall_s": wall if wall is not None else stop_when.t}}
    return run


def routine(rows, log, battery=True):
    return fe.FigureEight(open_rig=Rig(log), navigation_run=fake_run(rows, log),
                          open_battery=lambda: Battery(log) if battery else None)


def go(r, person, tmp_path, minutes=0.5):
    return h.run_routine(r, person.console(), tmp_path / "out", options={"minutes": minutes},
                         conditions_fn=no_conditions)


@pytest.mark.software
def test_one_continuous_run_judged_on_touches_time_laps_and_turns(tmp_path):
    log = []
    res = go(routine(lap_rows(laps=4), log), Person("", "0", "3", ""), tmp_path)   # 30 s: 3 laps and change
    assert res["verdict"] == h.PASS, res["criteria"]
    run = next(e for e in log if e[0] == "run")
    assert run == ("run", "attempt_01", 60.0, 8 * fe.MAX_LAPS, fe.ROUTE_LAP, True, False, True)
    assert "battery released" in log
    row = json.loads((tmp_path / "out" / "results.json").read_text())["rows"][0]
    assert (row["ended_by"], row["minutes"], row["laps_robot"], row["laps_seen"], row["touches"]) == \
        (fe.TIME_UP, 0.5, 3, 3, 0)
    assert (row["left_mean_deg"], row["right_mean_deg"], row["lap_s_mean"], row["turns_off_gyro"]) == \
        (-90.0, 90.0, 8.0, 0)
    assert row["battery_start_v"] == 11.99 and row["fps"] == pytest.approx(20.0, abs=0.1)
    with open(tmp_path / "out" / "attempt_01" / "intersections.csv") as f:
        assert len(list(csv.DictReader(f))) == 31                               # 30 s at 1 s an intersection
    with open(tmp_path / "out" / "attempt_01" / "laps.csv") as f:
        assert [r["lap"] for r in csv.DictReader(f)] == ["1", "2", "3"]


@pytest.mark.software
def test_a_run_ended_mid_turn_doesnt_count_that_lap(tmp_path):
    rows = lap_rows(laps=1)[:-10] + lap_rows(laps=1, t0=7.5, step0=8)[:5]      # the 8th turn never finishes
    w = fe.EightWatch(seconds=1e9)
    feed(w, rows)
    assert fe.FigureEight._row(w, {"ended_by": fe.TIME_UP, "run": {"wall_s": 8.0}}, 0, 0)["laps_robot"] == 0


@pytest.mark.software
@pytest.mark.parametrize("kw, answers, failing", [
    ({}, ("1", "3"), "touches"),
    ({"rows": lap_rows(laps=1)}, ("0", "1"), "ran the whole time"),
    ({}, ("0", "2"), "laps counted minus laps seen"),
    ({"rows": lap_rows(laps=4, turn_end=TURN_END_TIME)}, ("0", "3"), "turns not ended on the gyro"),
    ({"rows": lap_rows(laps=4, heading=lambda s, m: fe.EXPECTED_DEG[m] + (HEADING_TOLERANCE_DEG + 1) * (s == 6))},
     ("0", "3"), "turns off by over 20 deg"),
])
def test_each_criterion_fails_on_its_own(tmp_path, kw, answers, failing):
    log = []
    res = go(routine(kw.get("rows", lap_rows(laps=4)), log), Person("", *answers, ""), tmp_path)
    assert res["verdict"] == h.FAIL
    assert [c["name"] for c in res["criteria"] if not c["passed"]] == [failing]


@pytest.mark.software
def test_a_heading_right_at_the_tolerance_passes(tmp_path):
    rows = lap_rows(laps=4, heading=lambda s, m: fe.EXPECTED_DEG[m] + HEADING_TOLERANCE_DEG * (s == 6))
    assert go(routine(rows, []), Person("", "0", "3", ""), tmp_path)["verdict"] == h.PASS


@pytest.mark.software
def test_stopping_at_the_enter_prompt_closes_the_rig_and_the_battery(tmp_path):
    log = []
    res = go(routine([], log), Person("q"), tmp_path)
    assert res["stopped_early"]
    assert log[1:] == ["source closed", "sensors stopped", "motors stopped", "battery released"]
    log2 = []
    go(routine([], log2, battery=False), Person("q"), tmp_path / "b")
    assert log2[1:] == ["source closed", "sensors stopped", "motors stopped"]


@pytest.mark.software
def test_several_runs_are_judged_each(tmp_path):
    log = []
    r = routine(lap_rows(laps=4), log)
    r._run = fake_run(lap_rows(laps=4), log)
    res = h.run_routine(r, Person("", "0", "3", "", "", "0", "3", "").console(), tmp_path / "out", trials=2,
                        options={"minutes": 0.5}, conditions_fn=no_conditions)
    names = [c["name"] for c in res["criteria"]]
    assert len(names) == 10 and names[0] == "touches (run 1)" and names[5] == "touches (run 2)"
