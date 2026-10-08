"""
test_routine_recover_offset.py  --  src/routines/recover_offset.py

RecoverWatch over generated nav.csv rows: a recovery ended once held
within the band, the hold restarting when it leaves, the start, time to
recover, overshoot past the center, lane-lost time, the backstop; the
straight run kept to its time with its mean and max. The routine on a
fake rig and run: the straight run and each start, the run given the
scale and the straight route, the tester's start and end offsets, the
ruler deciding recovery, the maximum recoverable offset said, no scale
refused, the rig closed when the tester stops before driving; the
criteria one at a time and at their limits.

--software  A scripted tester, a fake rig and navigation run.
"""
import json

import pytest

import src.routines.harness as h
import src.routines.recover_offset as ro
from src.routines import ROUTINES
from src.tests.test_routines import Person, no_conditions

DT = 0.05


def glide(start, end, seconds, t0=0.0, status="vision"):
    """Rows moving the offset linearly from start to end over seconds."""
    n = int(round(seconds / DT))
    return [{"t": round(t0 + k * DT, 3), "lane_status": status,
             "lane_offset_cm": start + (end - start) * k / max(n - 1, 1)} for k in range(n)]


def recovery(start, overshoot=0.0, settle=0.0):
    """From start, past the center to -overshoot on the far side, then settle at `settle` for 2 s."""
    sign = 1 if start > 0 else -1
    rows = glide(start, -sign * overshoot, 1.0)
    return rows + glide(-sign * overshoot, settle, 0.5, t0=1.0) + glide(settle, settle, 2.0, t0=1.5)


def feed(watch, rows):
    for r in rows:
        why = watch(r)
        if why:
            return why
    return None


# =============================================================================
# RecoverWatch
# =============================================================================

@pytest.mark.software
def test_a_recovery_ends_once_held_in_the_band():
    w = ro.RecoverWatch(straight=False)
    assert feed(w, recovery(4.0, overshoot=1.0)) == ro.RECOVERED
    assert w.start_cm == 4.0 and w.overshoot_cm == pytest.approx(1.0)
    # in the band (|x| <= 1.5) from row 10 of 20 (4 -> -1 over 1 s): t = 0.5; held 1 s
    assert w.recover_s == 0.5 and w.t == pytest.approx(w.recover_s + ro.HOLD_S, abs=DT)


@pytest.mark.software
def test_leaving_the_band_restarts_the_hold_and_the_backstop_ends_it():
    rows = glide(3, 0, 0.5) + glide(0, 0, 0.5, t0=0.5) + glide(2, 2, 0.5, t0=1.0) + glide(0, 0, 1.5, t0=1.5)
    w = ro.RecoverWatch(straight=False)
    assert feed(w, rows) == ro.RECOVERED and w.recover_s == 1.5
    edge = ro.RecoverWatch(straight=False)
    assert feed(edge, glide(-1.5, -1.5, 1.5)) == ro.RECOVERED                 # the band's edge is in it
    stuck = ro.RecoverWatch(straight=False, max_s=2.0)
    assert feed(stuck, glide(4, 4, 3.0)) == ro.TIME_UP and stuck.recover_s is None and stuck.t == 2.0


@pytest.mark.software
def test_lost_frames_count_and_dont_start_or_hold():
    rows = glide(9, 9, 0.5, status="stale") + glide(3, 0, 0.5, t0=0.5) + [
        {"t": 1.0 + k * DT, "lane_status": "vision", "lane_offset_cm": None} for k in range(4)]
    w = ro.RecoverWatch(straight=False)
    feed(w, rows)
    assert w.start_cm == 3 and w.lane_lost_s == pytest.approx(0.45 + 4 * DT)     # stale rows, then the four without a value
    assert len(w.offsets) == 10


@pytest.mark.software
def test_the_straight_run_keeps_going_with_its_mean_and_max():
    rows = glide(0.5, -0.5, 6.0)
    w = ro.RecoverWatch(straight=True, max_s=5.0)
    assert feed(w, rows) == ro.TIME_UP and w.t == 5.0 and w.recover_s is None
    assert w.max_abs() == pytest.approx(0.5) and 0.2 < w.mean_abs() < 0.3
    assert w.overshoot_cm == pytest.approx(1 * 100 / 119 - 0.5)     # started right, at 5 s 0.34 left
    empty = ro.RecoverWatch(straight=True)
    assert empty.mean_abs() is None and empty.max_abs() is None


@pytest.mark.software
def test_a_start_on_the_center_has_no_overshoot():
    w = ro.RecoverWatch(straight=True)
    feed(w, [{"t": 0.0, "lane_status": "vision", "lane_offset_cm": 0.0},
             {"t": 0.05, "lane_status": "vision", "lane_offset_cm": -1.0}])
    assert w.overshoot_cm == 0.0


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


def fake_run(scripts, log):
    attempts = iter(scripts)

    def run(source, sensors, motor, nav, config, p3, out_dir, system, max_run_s, motors_on, render, stop_when):
        log.append(("run", out_dir.rsplit("/", 1)[-1], max_run_s, p3.cm_per_px, nav.progress.route.maneuvers,
                    motors_on, render))
        return {"ended_by": feed(stop_when, next(attempts)) or "time cap"}
    return run


STRAIGHT = glide(0.3, -0.3, 6.0)


def routine(scripts, log, scale=0.05):
    return ro.RecoverOffset(open_rig=Rig(log), navigation_run=fake_run(scripts, log), battery=lambda: 11.8,
                            cm_per_px=scale)


def answers(*pairs):
    out = []
    for start, end in pairs:
        out += ["", str(start), "", str(end), ""]       # placed, measured, drive, where it stopped, keep
    return out


def go(r, person, tmp_path, **options):
    return h.run_routine(r, person.console(), tmp_path / "out", options=options, conditions_fn=no_conditions)


def results(tmp_path):
    return json.loads((tmp_path / "out" / "results.json").read_text())


SHORT = {"positions": "0,2,-2,3,-3"}
OK_RUNS = [STRAIGHT, recovery(2), recovery(-2), recovery(3, 0.5), recovery(-3)]
OK_ANSWERS = answers((0, 0.2), (2, 0.3), (-2, -0.1), (3.1, 0.5), (-3, 0))


@pytest.mark.software
def test_the_straight_run_and_each_start_pass(tmp_path):
    log = []
    person = Person(*OK_ANSWERS)
    res = go(routine(OK_RUNS, log), person, tmp_path, **SHORT)
    assert res["verdict"] == h.PASS, res["criteria"]
    runs = [e for e in log if e[0] == "run"]
    assert runs[0] == ("run", "attempt_01", ro.STRAIGHT_S + 5.0, 0.05, ("straight",), True, False)
    assert runs[1][2] == ro.MAX_S + 5.0
    rows = results(tmp_path)["rows"]
    assert [r["planned_cm"] for r in rows] == [0, 2, -2, 3, -3]
    assert (rows[0]["max_abs_cm"], rows[0]["ended_by"], rows[0]["recover_s"]) == (0.3, ro.TIME_UP, None)
    assert (rows[3]["start_cm"], rows[3]["end_cm"], rows[3]["recovered"], rows[3]["robot_start_cm"]) == (3.1, 0.5, True, 3.0)
    assert rows[3]["overshoot_cm"] == 0.5 and rows[1]["ended_by"] == ro.RECOVERED and rows[1]["battery_v"] == 11.8
    assert results(tmp_path)["state"]["max_recoverable_cm"] == 3
    prompts = "\n".join(person.prompts)
    assert "centered in the lane" in prompts and prompts.index("2 cm RIGHT") < prompts.index("2 cm LEFT")
    assert "maximum recoverable offset: 3 cm" in person.said()


@pytest.mark.software
def test_the_ruler_decides_recovery(tmp_path):
    a = answers((0, 0), (2, 1.6), (-2, -0.1), (3, 0), (-3, 0))
    res = go(routine(OK_RUNS, []), Person(*a), tmp_path, **SHORT)
    assert results(tmp_path)["rows"][1]["recovered"] is False
    assert [c["name"] for c in res["criteria"] if not c["passed"]] == ["starts up to 3 cm not recovered"]
    assert results(tmp_path)["state"]["max_recoverable_cm"] is None


@pytest.mark.software
def test_a_miss_past_need_cm_only_limits_the_range(tmp_path):
    a = answers((0, 0), (2, 0), (-2, 0), (3, 0), (-3, 0), (4, 0), (-4, 2.5))
    runs = OK_RUNS + [recovery(4), glide(-4, -4, 7)]
    person = Person(*a)
    res = go(routine(runs, []), person, tmp_path, positions="0,2,-2,3,-3,4,-4")
    assert res["verdict"] == h.PASS and "maximum recoverable offset: 3 cm" in person.said()


@pytest.mark.software
def test_the_settings_band_need_and_times(tmp_path):
    log = []
    a = answers((0, 0), (2, 1.0))
    res = go(routine([STRAIGHT, recovery(2)], log), Person(*a), tmp_path, positions="0,2", band_cm=0.9,
             need_cm=2, max_s=3, straight_s=2)
    assert res["trials_planned"] == 2
    assert [c["name"] for c in res["criteria"] if not c["passed"]] == ["starts up to 2 cm not recovered"]
    assert [e[2] for e in log if e[0] == "run"] == [7.0, 8.0]


@pytest.mark.software
def test_no_scale_is_refused(tmp_path):
    with pytest.raises(ValueError, match="routine-lane-offset"):
        go(routine([], [], scale=None), Person(), tmp_path)
    assert results(tmp_path)["verdict"] == h.INCOMPLETE


@pytest.mark.software
def test_a_scale_from_the_settings_reaches_the_run(tmp_path):
    log = []
    go(routine([STRAIGHT], log, scale=None), Person(*answers((0, 0))), tmp_path, positions="0", cm_per_px=0.04)
    assert next(e for e in log if e[0] == "run")[3] == 0.04


@pytest.mark.software
def test_stopping_before_driving_closes_the_rig(tmp_path):
    log = []
    res = go(routine([], log), Person("", "2", "q"), tmp_path, positions="2")
    assert res["stopped_early"] and log[1:] == ["source closed", "sensors stopped", "motors stopped"]


@pytest.mark.software
def test_registered_and_planned():
    assert ROUTINES["recover-offset"] is ro.RecoverOffset and ro.RecoverOffset.trials == 9
    assert ro.RecoverOffset().plan({"positions": "0,1"}) == 2 and ro.BAND_CM == 1.5
    assert (ro.MAX_S, ro.STRAIGHT_S, ro.HOLD_S) == (6.0, 5.0, 1.0)


# =============================================================================
# The criteria
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("peak, verdict", [(1.5, h.PASS), (1.51, h.FAIL)])
def test_the_straight_runs_band(tmp_path, peak, verdict):
    runs = [glide(peak, peak, 6.0)] + OK_RUNS[1:]
    res = go(routine(runs, []), Person(*OK_ANSWERS), tmp_path, **SHORT)
    assert res["verdict"] == verdict
    if verdict == h.FAIL:
        assert [c["name"] for c in res["criteria"] if not c["passed"]] == ["straight run: farthest from the center"]


@pytest.mark.software
def test_a_straight_run_never_on_vision_fails(tmp_path):
    runs = [glide(0, 0, 6.0, status="stale")] + OK_RUNS[1:]
    res = go(routine(runs, []), Person(*OK_ANSWERS), tmp_path, **SHORT)
    assert res["criteria"][1]["value"] is None and res["verdict"] == h.FAIL


@pytest.mark.software
def test_without_a_straight_run_only_recovery_is_judged(tmp_path):
    res = go(routine(OK_RUNS[1:3], []), Person(*answers((2, 0), (-2, 1.5))), tmp_path, positions="2,-2")
    assert res["verdict"] == h.PASS and len(res["criteria"]) == 1      # 1.5 is still in the band


@pytest.mark.software
def test_the_worst_of_several_straight_runs_is_judged(tmp_path):
    runs = [glide(0.2, 0.2, 6.0), glide(1.8, 1.8, 6.0)]
    res = go(routine(runs, []), Person(*answers((0, 0), (0, 0))), tmp_path, positions="0,0")
    assert res["criteria"][1]["value"] == 1.8 and res["verdict"] == h.FAIL
