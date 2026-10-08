"""
test_routine_stop_distance.py  --  src/routines/stop_distance.py

StopWatch over scripted nav.csv rows: the last stop line before braking and
the speed on the last driving frame kept; stopped once braked for a stop
reason with both wheels still for STILL_S (jitter restarts it), or at once
when the stop sign's hold begins; other brakes ignored. The routine on a
fake rig and run: each trial opens the rig, waits for Enter, drives with the
watch, asks the gap, keeps one folder per attempt, closes the rig if the
tester stops at Enter; the criteria at their limits; a whole routine.

--software  A scripted tester, a fake rig and a fake navigation run.
"""
import pytest

import src.routines.harness as h
import src.routines.stop_distance as sd
from src.navigation.stop_sign import REASON_HOLD, REASON_STOPPING, STOPPED_CPS
from src.navigation.traffic_light import REASON_RED
from src.tests.test_routines import Person, no_conditions


def row(t, brake=0, reason="steer", line=None, cps=1166.0):
    return {"t": t, "brake": brake, "reason": reason, "stop_line_cm": "" if line is None else line,
            "left_cps": cps, "right_cps": cps}


def approach(reason=REASON_STOPPING, still_after=3, settle=8):
    """Driving at 1166 cps with the line closing in, then braking down to still."""
    rows = [row(i * 0.05, line=30.0 - 5 * i, cps=1166.0 - i) for i in range(5)]
    rows.append(row(0.25, line=None, cps=1100.0))                            # line under the robot
    t = 0.30
    for k in range(settle):
        rows.append(row(round(t, 2), 1, reason, cps=max(0.0, 800.0 - 400 * k) if k < still_after else 5.0))
        t += 0.05
    return rows


def feed(watch, rows):
    for r in rows:
        why = watch(r)
        if why:
            return why
    return None


# =============================================================================
# StopWatch
# =============================================================================

@pytest.mark.software
def test_the_watch_keeps_the_last_line_and_speed_before_braking_and_ends_when_still():
    w = sd.StopWatch()
    assert feed(w, approach()) == sd.STOPPED and w.stopped
    assert (w.reported_cm, w.speed_cps, w.reason, w.brake_t) == (10.0, 1100.0, REASON_STOPPING, 0.3)


@pytest.mark.software
def test_still_means_both_wheels_at_or_under_the_stop_signs_threshold_for_still_s():
    w = sd.StopWatch()
    rows = [row(0.0, line=10.0)] + [row(round(0.05 * k, 2), 1, REASON_RED, cps=STOPPED_CPS) for k in range(1, 6)]
    # each still row counts the 50 ms before it: 0.25 s by t=0.25, 0.30 s at t=0.30
    assert feed(w, rows) is None
    assert w(row(0.30, 1, REASON_RED, cps=STOPPED_CPS)) == sd.STOPPED
    jitter = sd.StopWatch()
    rows = [row(0.0)] + [row(round(0.05 * k, 2), 1, REASON_RED, cps=(STOPPED_CPS + 1) if k == 4 else 0.0)
                         for k in range(1, 9)]
    assert feed(jitter, rows) is None                                        # restarted at t=0.20


@pytest.mark.software
def test_the_stop_signs_hold_ends_it_at_once_and_other_brakes_are_ignored():
    w = sd.StopWatch()
    assert w(row(0.0, line=8.0)) is None
    assert w(row(0.05, 1, "lane_lost", cps=0.0)) is None and w.brake_t is None
    assert w(row(0.10, 1, REASON_HOLD, cps=300.0)) == sd.STOPPED and w.reason == REASON_HOLD
    never = sd.StopWatch()
    assert feed(never, [row(k * 0.05, line=5.0) for k in range(40)]) is None and not never.stopped


@pytest.mark.software
def test_a_braked_row_without_encoders_reads_still():
    w = sd.StopWatch()
    rows = [row(0.0)] + [{"t": round(0.05 * k, 2), "brake": "1", "reason": REASON_STOPPING, "stop_line_cm": "",
                          "left_cps": "", "right_cps": ""} for k in range(1, 9)]
    assert feed(w, rows) == sd.STOPPED


# =============================================================================
# The routine
# =============================================================================

class Rig:
    def __init__(self, log):
        self.log = log
        self.source = type("S", (), {"close": lambda s: log.append("source closed")})()
        self.sensors = type("Se", (), {"stop": lambda s: log.append("sensors stopped")})()
        self.motor = type("M", (), {"stop": lambda s: log.append("motors stopped")})()

    def __call__(self, camera, motors, button):
        self.log.append(("open", camera, motors, button))
        return self.source, self.sensors, self.motor, None


def fake_run(scripts, log):
    """navigation_linker.run stand-in: feeds each attempt's rows to stop_when; ended_by its answer or the cap."""
    attempts = iter(scripts)

    def run(source, sensors, motor, nav, config, p3, out_dir, system, max_run_s, motors_on, render, stop_when):
        log.append(("run", out_dir.rsplit("/", 1)[-1], max_run_s, motors_on, render))
        why = feed(stop_when, next(attempts))
        return {"ended_by": why or "time cap"}
    return run


def routine(scripts, log, volts=(11.9, 11.8, 11.7, 11.6, 11.5)):
    v = iter(volts)
    return sd.StopDistance(open_rig=Rig(log), navigation_run=fake_run(scripts, log), battery=lambda: next(v))


def go(r, person, tmp_path, trials=3, options=None):
    return h.run_routine(r, person.console(), tmp_path / "out", trials=trials, options=options,
                         conditions_fn=no_conditions)


@pytest.mark.software
def test_each_trial_drives_to_the_line_and_keeps_the_tape_beside_the_report(tmp_path):
    log = []
    res = go(routine([approach(), approach(), approach()], log), Person("", "4.0", "", "", "4.5", "", "", "3.5", ""),
             tmp_path)
    assert res["verdict"] == h.PASS
    assert log[0] == ("open", True, True, False)
    assert [e for e in log if isinstance(e, tuple) and e[0] == "run"] == [
        ("run", f"attempt_0{k}", sd.TRIAL_MAX_S, True, False) for k in (1, 2, 3)]
    rows = __import__("json").loads((tmp_path / "out" / "results.json").read_text())["rows"]
    assert [(r["gap_cm"], r["reported_cm"], r["speed_cps"], r["battery_v"], r["reason"]) for r in rows] == [
        (4.0, 10.0, 1100.0, 11.9, REASON_STOPPING), (4.5, 10.0, 1100.0, 11.8, REASON_STOPPING),
        (3.5, 10.0, 1100.0, 11.7, REASON_STOPPING)]
    assert res["options"] == {"start_cm": sd.START_GAP_CM, "max_s": sd.TRIAL_MAX_S}


@pytest.mark.software
def test_a_redo_gets_its_own_folder_and_the_backstop_is_a_setting(tmp_path):
    log = []
    res = go(routine([approach(), approach()], log), Person("", "4.0", "r", "", "4.2", ""), tmp_path, trials=1,
             options={"max_s": 12})
    runs = [e[1:3] for e in log if isinstance(e, tuple) and e[0] == "run"]
    assert runs == [("attempt_01", 12.0), ("attempt_02", 12.0)] and res["trials_kept"] == 1


@pytest.mark.software
def test_stopping_at_the_enter_prompt_closes_the_rig(tmp_path):
    log = []
    res = go(routine([], log), Person("q"), tmp_path)
    assert res["stopped_early"] and log[1:] == ["source closed", "sensors stopped", "motors stopped"]


@pytest.mark.software
def test_a_run_that_never_stops_says_so_and_fails_the_stop_criterion(tmp_path):
    log = []
    p = Person("", "-3.0", "")
    res = go(routine([[row(k * 0.05) for k in range(10)]], log), p, tmp_path, trials=1)
    assert "it never braked for the line" in p.said() and "ended by time cap" in p.said()
    assert res["verdict"] == h.FAIL and not res["criteria"][0]["passed"]


@pytest.mark.software
def test_the_criteria_at_their_limits():
    r = sd.StopDistance()

    def rows(*gaps, ended=sd.STOPPED):
        return [{"gap_cm": g, "ended_by": ended} for g in gaps]
    lo, hi = sd.GAP_RANGE_CM
    assert all(c["passed"] for c in r.judge(rows(lo, lo + sd.MAX_SPREAD_CM)))
    assert [c["passed"] for c in r.judge(rows(lo, lo + sd.MAX_SPREAD_CM + 0.1))] == [True, True, False]
    assert [c["passed"] for c in r.judge(rows(hi + 0.5, hi + 0.5))] == [True, False, True]
    assert r.judge(rows(0.0, 0.0))[0]["passed"] and not r.judge(rows(-0.1, 4.0))[0]["passed"]
    assert not r.judge(rows(4.0, ended="time cap"))[0]["passed"]
