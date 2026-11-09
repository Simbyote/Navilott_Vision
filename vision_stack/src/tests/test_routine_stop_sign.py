"""
test_routine_stop_sign.py  --  src/routines/stop_sign.py

SignWatch over scripted nav.csv rows: the sign vote seen, each stop-sign
brake counted, the hold timed from its first frame to the first frame
after it, driving on, the run ended a second into the exit stage; another
rule's brake ignored. The routine on a fake rig and run: sign trials then
no-sign ones, each prompt saying which; each criterion on its own.

--software  A scripted tester, a fake rig and a fake navigation run.
"""
import json

import pytest

import src.routines.harness as h
import src.routines.stop_sign as ss
from src.navigation.intersection import STAGE_EXIT, STAGE_TO_LINE
from src.navigation.navigation import RULE_INTERSECTION, RULE_LANE_KEEPING, RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT
from src.navigation.stop_sign import REASON_HOLD, REASON_STOPPING, STOP_SIGN_HOLD_TIME_MS
from src.tests.test_routine_stop_distance import Rig, feed
from src.tests.test_routines import Person, no_conditions

HOLD_S = STOP_SIGN_HOLD_TIME_MS / 1000


def row(t, rule=RULE_LANE_KEEPING, reason="steer", brake=0, sign=0, stage=""):
    return {"t": round(t, 2), "rule": rule, "reason": reason, "brake": brake, "stop_sign": sign, "stage": stage}


def frames(start, seconds, **kw):
    return [row(start + k * 0.05, **kw) for k in range(int(round(seconds / 0.05)))]


def with_sign(hold_s=HOLD_S, seen=True):
    """Approach with the sign voted, a stop, the hold, then across."""
    rows = frames(0.0, 1.0, sign=int(seen)) + frames(1.0, 0.5, rule=RULE_INTERSECTION, stage=STAGE_TO_LINE)
    rows += frames(1.5, 0.3, rule=RULE_STOP_SIGN, reason=REASON_STOPPING, brake=1, stage=STAGE_TO_LINE)
    rows += frames(1.8, hold_s, rule=RULE_STOP_SIGN, reason=REASON_HOLD, brake=1, stage=STAGE_TO_LINE)
    return rows + frames(1.8 + hold_s, 1.5, rule=RULE_INTERSECTION, stage=STAGE_EXIT)


def without_sign():
    return frames(0.0, 1.0) + frames(1.0, 0.5, rule=RULE_INTERSECTION, stage=STAGE_TO_LINE) \
        + frames(1.5, 1.5, rule=RULE_INTERSECTION, stage=STAGE_EXIT)


@pytest.mark.software
def test_the_watch_times_the_hold_and_ends_a_second_into_the_exit():
    w = ss.SignWatch()
    assert feed(w, with_sign()) == ss.ACROSS
    assert (w.sign_seen, w.stops, w.resumed, w.across) == (True, 1, True, True)
    assert w.hold_s == pytest.approx(HOLD_S, abs=0.051)
    assert w._exit_t == pytest.approx(1.8 + HOLD_S)                  # ended EXIT_DRIVE_S after this


@pytest.mark.software
def test_no_sign_crosses_without_a_stop_and_other_brakes_dont_count():
    w = ss.SignWatch()
    assert feed(w, without_sign()) == ss.ACROSS
    assert (w.sign_seen, w.stops, w.hold_s, w.resumed) == (False, 0, None, False)
    red = ss.SignWatch()
    rows = frames(0.0, 1.0) + frames(1.0, 2.0, rule=RULE_TRAFFIC_LIGHT, reason="red_light", brake=1)
    assert feed(red, rows + frames(3.0, 1.5, rule=RULE_INTERSECTION, stage=STAGE_EXIT)) == ss.ACROSS
    assert red.stops == 0


@pytest.mark.software
def test_a_stop_that_never_lets_go_has_no_hold_and_never_ends():
    w = ss.SignWatch()
    rows = frames(0.0, 1.0, sign=1) + frames(1.0, 4.0, rule=RULE_STOP_SIGN, reason=REASON_HOLD, brake=1)
    assert feed(w, rows) is None and (w.stops, w.hold_s, w.resumed, w.across) == (1, None, False, False)


def fake_run(scripts, log):
    attempts = iter(scripts)

    def run(source, sensors, motor, nav, config, p3, out_dir, system, max_run_s, motors_on, render, stop_when):
        log.append(("run", out_dir.rsplit("/", 1)[-1], max_run_s))
        return {"ended_by": feed(stop_when, next(attempts)) or "time cap"}
    return run


def go(scripts, person, tmp_path, **options):
    log = []
    r = ss.StopSign(open_rig=Rig(log), navigation_run=fake_run(scripts, log), battery=lambda: 11.8)
    return h.run_routine(r, person.console(), tmp_path / "out", options=options, conditions_fn=no_conditions), log


@pytest.mark.software
def test_sign_trials_then_no_sign_ones_pass(tmp_path):
    person = Person(*[""] * 6)
    res, log = go([with_sign(), with_sign(2.04), without_sign()], person, tmp_path, signs=2, no_signs=1)
    assert res["verdict"] == h.PASS, res["criteria"]
    rows = json.loads((tmp_path / "out" / "results.json").read_text())["rows"]
    assert [(r["sign"], r["stops"], r["across"], r["ended_by"]) for r in rows] == [
        (ss.PRESENT, 1, True, ss.ACROSS), (ss.PRESENT, 1, True, ss.ACROSS), (ss.ABSENT, 0, True, ss.ACROSS)]
    prompts = "\n".join(person.prompts)
    assert prompts.count("stop sign POSTED") == 2 and prompts.count("TAKEN AWAY") == 1
    assert [e[1] for e in log if isinstance(e, tuple) and e[0] == "run"] == ["attempt_01", "attempt_02", "attempt_03"]


@pytest.mark.software
def test_the_plan_follows_the_settings():
    assert ss.StopSign().plan({}) == ss.SIGN_TRIALS + ss.NO_SIGN_TRIALS
    assert ss.StopSign().plan({"signs": "4", "no_signs": 0}) == 4


def sign(stops=1, hold=HOLD_S, resumed=True, across=True):
    return {"sign": ss.PRESENT, "stops": stops, "hold_s": hold, "resumed": resumed, "across": across}


def none(stops=0, across=True):
    return {"sign": ss.ABSENT, "stops": stops, "hold_s": None, "resumed": stops > 0, "across": across}


@pytest.mark.software
def test_each_criterion_on_its_own():
    r = ss.StopSign()
    assert all(c["passed"] for c in r.judge([sign(), sign(hold=HOLD_S + ss.HOLD_TOLERANCE_S), none()]))

    def failed(rows):
        return [c["name"] for c in r.judge(rows) if not c["passed"]]
    assert failed([sign(stops=0, hold=None, resumed=False)]) == [
        "sign trials it didn't stop at", f"worst hold off {HOLD_S:g} s", "sign trials it didn't drive on across"]
    assert failed([sign(hold=HOLD_S - 0.31)]) == [f"worst hold off {HOLD_S:g} s"]
    assert failed([sign(across=False)]) == ["sign trials it didn't drive on across"]
    assert failed([none(stops=1)]) == ["stops for a sign that wasn't there"]
    assert failed([none(across=False)]) == ["no-sign trials it didn't drive across"]
    assert [c["name"] for c in r.judge([none()])] == ["stops for a sign that wasn't there",
                                                     "no-sign trials it didn't drive across"]
