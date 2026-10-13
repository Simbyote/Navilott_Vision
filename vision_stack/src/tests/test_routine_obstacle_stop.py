"""
test_routine_obstacle_stop.py  --  src/routines/obstacle_stop.py

ObstacleWatch over scripted nav.csv rows: the distance and speed on the
last frame before braking (clipped kept as clipped), stopped once still,
the clear wait and the drive-on timed, the run ended a second after
driving on; a brake for anything but the obstacle ignored; a clear lane
ended after clear_s, each obstacle brake on it counted. The routine on a
fake rig and run: object trials then clear ones, the gap asked only with
the object; each criterion on its own.

--software  A scripted tester, a fake rig and a fake navigation run.
"""
import json

import pytest

import src.routines.harness as h
import src.routines.obstacle_stop as os_
from src.navigation.navigation import RULE_LANE_KEEPING, RULE_OBSTACLE, RULE_STOP_SIGN
from src.navigation.obstacle import OBSTACLE_CLEAR_MS, REASON_CLEARING, REASON_OBSTACLE
from src.tests.test_routine_stop_distance import Rig, feed
from src.tests.test_routines import Person, no_conditions


def row(t, rule=RULE_LANE_KEEPING, reason="steer", brake=0, cm=None, cps=1200.0):
    return {"t": round(t, 2), "rule": rule, "reason": reason, "brake": brake, "obstacle_cm": "" if cm is None else cm,
            "left_cps": cps, "right_cps": cps}


def stop_for_object(clear_after=1.0, wait_s=OBSTACLE_CLEAR_MS / 1000, drive_on=1.2, near="15.5"):
    """Driving up to the object, braking to still, the object lifted after clear_after, driving on."""
    rows = [row(k * 0.05, cm=30.0 - 3 * k) for k in range(5)] + [row(0.25, cm=near, cps=1180.0)]
    t = 0.30
    while t < 0.30 + clear_after:
        rows.append(row(t, RULE_OBSTACLE, REASON_OBSTACLE, 1, cm=12.0, cps=0.0 if t > 0.4 else 600.0))
        t += 0.05
    end = t + wait_s
    while t < end - 1e-9:
        rows.append(row(t, RULE_OBSTACLE, REASON_CLEARING, 1, cps=0.0))
        t += 0.05
    while t < end + drive_on:
        rows.append(row(t))
        t += 0.05
    return rows


def lane(seconds, start=0.0, **kw):
    return [row(start + k * 0.05, **kw) for k in range(int(seconds / 0.05) + 1)]


# =============================================================================
# ObstacleWatch
# =============================================================================

@pytest.mark.software
def test_the_watch_follows_a_stop_the_wait_and_driving_on():
    w = os_.ObstacleWatch(True)
    assert feed(w, stop_for_object()) == os_.RESUMED
    assert (w.brakes, w.stopped, w.reported_cm, w.clipped, w.speed_cps) == (1, True, 15.5, False, 1180.0)
    assert w.resume_s == pytest.approx(OBSTACLE_CLEAR_MS / 1000, abs=0.051)
    assert w.resumed_t is not None and w.clear_t is not None


@pytest.mark.software
def test_a_clipped_obstacle_has_no_distance_and_a_never_stopping_run_reads_so():
    w = os_.ObstacleWatch(True)
    feed(w, stop_for_object(near="view bottom"))
    assert (w.reported_cm, w.clipped) == (None, True)
    never = os_.ObstacleWatch(True)
    assert feed(never, lane(3.0, cm=20.0)) is None
    assert (never.brakes, never.stopped, never.resume_s) == (0, False, None)


@pytest.mark.software
def test_only_the_obstacle_rules_brakes_count():
    w = os_.ObstacleWatch(False, clear_s=2.0)
    rows = lane(0.5) + [row(0.55 + k * 0.05, RULE_STOP_SIGN, "stop_sign_hold", 1) for k in range(5)] + lane(2.0, 0.8)
    assert feed(w, rows) == os_.CLEAR_DONE and w.brakes == 0


@pytest.mark.software
def test_a_clear_lane_counts_every_obstacle_brake_and_ends_after_clear_s():
    w = os_.ObstacleWatch(False, clear_s=3.0)
    rows = (lane(0.5) + [row(0.55, RULE_OBSTACLE, REASON_OBSTACLE, 1)] + lane(0.5, 0.6)
            + [row(1.15, RULE_OBSTACLE, REASON_OBSTACLE, 1), row(1.2, RULE_OBSTACLE, REASON_CLEARING, 1)]
            + lane(2.0, 1.25))
    assert feed(w, rows) == os_.CLEAR_DONE and w.brakes == 2
    assert w._last_t == pytest.approx(3.0)


# =============================================================================
# The routine
# =============================================================================

def fake_run(scripts, log):
    attempts = iter(scripts)

    def run(source, sensors, motor, nav, config, p3, out_dir, system, max_run_s, motors_on, render, stop_when):
        log.append(("run", out_dir.rsplit("/", 1)[-1], max_run_s, stop_when.with_object))
        return {"ended_by": feed(stop_when, next(attempts)) or "time cap"}
    return run


def go(scripts, person, tmp_path, **options):
    log = []
    r = os_.ObstacleStop(open_rig=Rig(log), navigation_run=fake_run(scripts, log), battery=lambda: 11.8)
    res = h.run_routine(r, person.console(), tmp_path / "out", options=options, conditions_fn=no_conditions)
    return res, log


@pytest.mark.software
def test_object_trials_then_clear_ones_pass(tmp_path):
    person = Person("", "10.5", "", "", "9.0", "", "", "", "")      # object: Enter, gap, keep; clear: Enter, keep
    res, log = go([stop_for_object(), stop_for_object(), lane(6.5), ], person, tmp_path, objects=2, clear=1,
                  clear_s=6)
    assert res["verdict"] == h.PASS, res["criteria"]
    assert [e[3] for e in log if isinstance(e, tuple) and e[0] == "run"] == [True, True, False]
    rows = json.loads((tmp_path / "out" / "results.json").read_text())["rows"]
    assert [(r["state"], r["gap_cm"], r["brakes"], r["stopped"]) for r in rows] == [
        (os_.OBJECT, 10.5, 1, True), (os_.OBJECT, 9.0, 1, True), (os_.CLEAR, None, 0, False)]
    assert [r["ended_by"] for r in rows] == [os_.RESUMED, os_.RESUMED, os_.CLEAR_DONE]
    prompts = "\n".join(person.prompts)
    assert prompts.count("the object in the lane") == 2 and prompts.count("the lane CLEAR") == 1
    assert prompts.count("Gap from the front edge") == 2


@pytest.mark.software
def test_the_plan_follows_the_settings():
    assert os_.ObstacleStop().plan({}) == os_.OBJECT_TRIALS + os_.CLEAR_TRIALS
    assert os_.ObstacleStop().plan({"objects": 3, "clear": "0"}) == 3


def obj(gap=10.0, stopped=True, resume=1.0):
    return {"state": os_.OBJECT, "gap_cm": gap, "stopped": stopped, "resume_s": resume, "brakes": 1}


def clr(brakes=0):
    return {"state": os_.CLEAR, "gap_cm": None, "stopped": False, "resume_s": None, "brakes": brakes}


@pytest.mark.software
def test_each_criterion_on_its_own():
    r = os_.ObstacleStop()
    assert all(c["passed"] for c in r.judge([obj(), obj(14.0), clr()]))

    def failed(rows):
        return [c["name"].split(" (")[0] for c in r.judge(rows) if not c["passed"]]
    assert failed([obj(stopped=False), clr()]) == ["object trials it didn't stop for"]
    assert failed([obj(0.0)]) == ["object trials it didn't stop for", "gaps outside 5..15 cm"]      # touched
    assert failed([obj(16.0)]) == ["gaps outside 5..15 cm"]
    assert failed([obj(resume=None)]) == ["object trials it didn't drive on from", "slowest drive-on after the object went"]
    limit = OBSTACLE_CLEAR_MS / 1000 + os_.RESUME_SLACK_S
    assert failed([obj(resume=limit)]) == [] and failed([obj(resume=limit + 0.01)]) == [
        "slowest drive-on after the object went"]
    assert failed([obj(), clr(1)]) == ["false brakes on a clear lane"]
    assert [c["name"] for c in r.judge([clr()])] == ["false brakes on a clear lane"]
