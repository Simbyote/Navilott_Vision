"""
test_routine_course_run.py  --  src/routines/course_run.py

CourseWatch over generated nav.csv rows: the highest route step, stop
sign holds and red-light waits counted once each, lane-lost time in lane
keeping only, contract brakes, the pack; it never ends a run. The routine
on a fake rig, battery and run: the route file loaded and said before the
rig opens (a bad one stops it), each run with the start button, the
route, the battery and the cap, the tester's answers, completed only when
the tester and the robot agree, the stopwatch against the robot's time,
best and mean of completed runs, the rig, button and battery closed when
the tester stops at Enter; each criterion on its own.

--software  A scripted tester, a fake rig, battery and navigation run.
"""
import json

import pytest

import src.routines.course_run as cr
import src.routines.harness as h
from src.navigation.route import RouteError
from src.navigation.stop_sign import REASON_HOLD
from src.navigation.traffic_light import REASON_RED
from src.navigation_linker import END_COURSE, END_EARLY
from src.routines import ROUTINES
from src.tests.test_routines import Person, no_conditions

DT = 0.05
ROUTE = ["right", "left", "straight"]


def course(steps=3, hold_at=None, red_at=None, lost=0, contract=0, volts=12.0):
    """Rows: per step 10 lane-keeping frames (some off vision), a stop sign hold or red wait where asked."""
    out, t = [], 0.0
    for k in range(1, steps + 1):
        label = f"{k}/{len(ROUTE)} {ROUTE[(k - 1) % len(ROUTE)]}"
        for f in range(10):
            reason = "steer"
            if k == hold_at and 3 <= f < 6:
                reason = REASON_HOLD
            if k == red_at and 6 <= f < 9:
                reason = REASON_RED
            if contract and k == 1 and f == 0:
                reason = "contract"
            status = "hold" if f < lost else "vision"
            out.append({"t": round(t, 2), "rule": "lane_keeping", "step": label, "lane_status": status,
                        "reason": reason, "battery_v": round(volts - t * 0.01, 3)})
            t += DT
    return out


def feed(watch, rows):
    for r in rows:
        assert watch(r) is None


# =============================================================================
# CourseWatch
# =============================================================================

@pytest.mark.software
def test_the_watch_counts_steps_stops_and_waits():
    w = cr.CourseWatch()
    feed(w, course(hold_at=1, red_at=2))
    assert (w.steps, w.holds, w.reds, w.frames, w.contract, w.lane_lost_s) == (3, 1, 1, 30, 0, 0.0)
    assert w.volts[0] == 12.0 and w.volts[-1] == pytest.approx(12.0 - 29 * DT * 0.01, abs=1e-3)


@pytest.mark.software
def test_two_stops_count_twice_and_lost_time_is_lane_keeping_only():
    w = cr.CourseWatch()
    rows = course(hold_at=1, lost=2, contract=1) + [
        {**r, "t": r["t"] + 10.0} for r in course(steps=1, hold_at=1)]
    rows.insert(5, {"t": rows[4]["t"], "rule": "intersection", "step": "1/3 right", "lane_status": "stale",
                    "reason": REASON_HOLD})
    feed(w, rows)
    assert w.holds == 2 and w.contract == 1
    # each step's first two frames off vision: the gap into each counts (the run's first frame has none);
    # the 10 s gap leads into a frame on vision, and the intersection row isn't lane keeping
    assert w.lane_lost_s == pytest.approx(5 * DT)
    assert w.steps == 3


@pytest.mark.software
def test_odd_rows_are_harmless():
    w = cr.CourseWatch()
    feed(w, [{"t": 0.0, "step": None, "reason": None, "battery_v": ""}, {"t": 0.1, "step": "x/3"}])
    assert (w.steps, w.holds, w.volts) == (0, 0, [])


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
                type("M", (), {"stop": lambda s: log.append("motors stopped")})(),
                type("Sy", (), {"cleanup": lambda s: log.append("button released")})())


class Battery:
    def __init__(self, log):
        self.log = log

    def cleanup(self):
        self.log.append("battery released")


def fake_run(scripts, log):
    attempts = iter(scripts)

    def run(source, sensors, motor, nav, config, p3, out_dir, system, max_run_s, motors_on, render, stop_when,
            battery):
        rows, ended, wall = next(attempts)
        log.append(("run", out_dir.rsplit("/", 1)[-1], nav.progress.route.maneuvers, system is not None, max_run_s,
                    motors_on, render, battery is not None))
        feed(stop_when, rows)
        return {"ended_by": ended, "run": {"wall_s": wall}}
    return run


@pytest.fixture
def route_file(tmp_path):
    p = tmp_path / "route.json"
    p.write_text(json.dumps({"maneuvers": ROUTE, "finish": "edge"}))
    return str(p)


def go(scripts, answers, tmp_path, route_file, log=None, battery=True, trials=None, **options):
    log = [] if log is None else log
    r = cr.CourseRun(open_rig=Rig(log), navigation_run=fake_run(scripts, log),
                     open_battery=lambda: Battery(log) if battery else None)
    person = Person(*answers)
    res = h.run_routine(r, person.console(), tmp_path / "out", trials=trials,
                        options={"route": route_file, **options}, conditions_fn=no_conditions)
    return res, person


def results(tmp_path):
    return json.loads((tmp_path / "out" / "results.json").read_text())


DONE = (course(hold_at=1, red_at=2), END_COURSE, 1.5)


def said(completed="y", stopwatch="1.6", touches="0", violations="0"):
    return ["", completed, stopwatch, touches, violations, ""]


@pytest.mark.software
def test_three_clean_runs_pass(tmp_path, route_file):
    log = []
    res, person = go([DONE, (course(), END_COURSE, 1.4), DONE], said() + said(stopwatch="1.3") + said(), tmp_path,
                     route_file, log)
    assert res["verdict"] == h.PASS, res["criteria"]
    assert log[0] == ("open", True, True, True)
    assert log[1] == ("run", "attempt_01", tuple(ROUTE), True, cr.MAX_S, True, False, True)
    assert "battery released" in log
    row = results(tmp_path)["rows"][0]
    assert (row["completed"], row["ended_by"], row["stopwatch_s"], row["robot_s"], row["clock_diff_s"]) == \
        (True, END_COURSE, 1.6, 1.5, -0.1)
    assert (row["steps_done"], row["steps_planned"], row["holds"], row["reds"], row["fps"]) == (3, 3, 1, 1, 20.0)
    assert row["battery_start_v"] == 12.0
    state = results(tmp_path)["state"]
    assert (state["best_s"], state["mean_s"]) == (1.3, 1.5)
    assert "route " in person.said() and "right, left, straight (3 steps), finish edge" in person.said()
    assert "best 1.3 s, mean 1.5 s" in person.said() and len(res["criteria"]) == 9


@pytest.mark.software
def test_completed_needs_the_tester_and_the_robot(tmp_path, route_file):
    res, _ = go([(course(steps=2), END_EARLY, 1.0)], said(stopwatch="0"), tmp_path, route_file, trials=1)
    row = results(tmp_path)["rows"][0]
    assert row["completed"] is False and row["clock_diff_s"] is None and row["steps_done"] == 2
    res, _ = go([DONE], said(completed="n"), tmp_path / "b", route_file)
    assert results(tmp_path / "b")["rows"][0]["completed"] is False


@pytest.mark.software
def test_only_completed_runs_have_times(tmp_path, route_file):
    _, person = go([(course(steps=1), END_EARLY, 0.5)] + [DONE] * 2,
                   said(stopwatch="0.6") + said(stopwatch="2") + said(completed="n", stopwatch="1"),
                   tmp_path, route_file)
    state = results(tmp_path)["state"]
    assert (state["best_s"], state["mean_s"]) == (2.0, 2.0) and "completed runs: 1" in person.said()
    _, person = go([(course(steps=1), END_EARLY, 0.5)], said(completed="n", stopwatch="0"), tmp_path / "b", route_file)
    assert "best_s" not in results(tmp_path / "b")["state"] and "completed runs" not in person.said()


@pytest.mark.software
def test_no_robot_time(tmp_path, route_file):
    go([(course(), END_COURSE, None)], said(), tmp_path, route_file)
    row = results(tmp_path)["rows"][0]
    assert row["robot_s"] is None and row["clock_diff_s"] is None and row["fps"] is None


@pytest.mark.software
def test_a_bad_route_stops_before_the_rig_opens(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"maneuvers": ["sideways"], "finish": "edge"}))
    log = []
    with pytest.raises(RouteError):
        go([], [], tmp_path, str(bad), log)
    assert log == []


@pytest.mark.software
def test_the_cap_is_a_setting(tmp_path, route_file):
    log = []
    go([DONE], said(), tmp_path, route_file, log, max_s=120)
    assert log[1][4] == 120.0


@pytest.mark.software
def test_stopping_at_enter_closes_everything(tmp_path, route_file):
    log = []
    res, _ = go([], ["q"], tmp_path, route_file, log)
    assert res["stopped_early"]
    assert log[1:] == ["source closed", "sensors stopped", "motors stopped", "button released", "battery released"]
    log2 = []
    go([], ["q"], tmp_path / "b", route_file, log2, battery=False)
    assert log2[1:] == ["source closed", "sensors stopped", "motors stopped", "button released"]


@pytest.mark.software
def test_registered():
    assert ROUTINES["course-run"] is cr.CourseRun and cr.CourseRun.trials == 3
    assert cr.FINISHED == END_COURSE and cr.MAX_S == 300.0


# =============================================================================
# The criteria
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("answers, failing", [
    (said(completed="n"), ["completed the course"]),
    (said(touches="1"), ["touches"]),
    (said(violations="1"), ["stop signs or red lights run"]),
])
def test_each_criterion_fails_on_its_own(tmp_path, route_file, answers, failing):
    res, _ = go([DONE], answers, tmp_path, route_file, trials=1)
    assert [c["name"] for c in res["criteria"] if not c["passed"]] == failing


@pytest.mark.software
def test_reds_and_holds_apart_and_only_lane_keeping_lost():
    w = cr.CourseWatch()
    feed(w, course(red_at=1) + [{"t": 5.0, "rule": "intersection", "step": "3/3", "lane_status": "stale"},
                                {"t": 6.0, "rule": "intersection", "step": "3/3", "lane_status": "stale"}])
    assert (w.reds, w.holds, w.lane_lost_s) == (1, 0, 0.0)


@pytest.mark.software
def test_a_completed_run_with_no_stopwatch_has_no_time_and_fps_rounds_to_tenths(tmp_path, route_file):
    go([(course(), END_COURSE, 1.6)], said(stopwatch="0"), tmp_path, route_file, trials=1)
    assert "best_s" not in results(tmp_path)["state"]
    assert results(tmp_path)["rows"][0]["fps"] == 18.8                       # 30 frames / 1.6 s
