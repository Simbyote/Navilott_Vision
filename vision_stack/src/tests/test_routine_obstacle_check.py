"""
test_routine_obstacle_check.py  --  src/routines/obstacle_check.py, camera_look's obstacle fields

camera_look: a frame's obstacle (Phase 2's and the vote's) in the Seen
dict. obstacle_check: the rule's brake decision on a frame; a look read
OK, MISSED, OFF or WRONG, its median distance and error, clipped frames
not measured; the routine on fake eyes: one trial per gap nearest first
then the clear trials, planned from the settings, a JPEG per attempt, the
range in the summary; each criterion on its own, and the brake decision
not judged next to the stop distance. Laterals: every gap at every offset,
the offset in the prompt and the file name, only objects in the path
judged, and the seen width per gap.

--software  A scripted tester and fake eyes.
"""
import json
from types import SimpleNamespace

import pytest

import src.routines.harness as h
import src.routines.obstacle_check as oc
from src.config import MEASURED_ESTIMATION
from src.navigation.obstacle import OBSTACLE_STOP_CM
from src.routines import ROUTINES
from src.routines.camera_look import seen
from src.tests.test_routine_detect_range import FakeEyes
from src.tests.test_routines import Person, no_conditions


def frame(cm=None, seen_=False, clipped=False, voted=False, voted_cm=None, voted_clipped=False):
    return {"obstacle": seen_, "obstacle_cm": cm, "obstacle_clipped": clipped, "obstacle_voted": voted,
            "obstacle_voted_cm": voted_cm, "obstacle_voted_clipped": voted_clipped}


def at(cm, n=40):
    """An object seen and voted at cm on every frame."""
    return [frame(cm, True, voted=True, voted_cm=cm) for _ in range(n)]


def clear(n=40):
    return [frame() for _ in range(n)]


@pytest.mark.software
def test_seen_carries_phase_2s_obstacle_and_the_vote():
    ob = SimpleNamespace(detected=True, distance_cm=18.5, clipped=False)
    res = SimpleNamespace(
        chain=SimpleNamespace(phase2=SimpleNamespace(detections=[], obstacle_results=[ob]),
                              offset=SimpleNamespace(offset=0.0, mode="two", lane_width_px=300)),
        packet=SimpleNamespace(drive_state="go", stop_sign_detected=False, lane_offset=0.0, lane_status="vision",
                               obstacle_detected=True, obstacle_distance_cm=18.0, obstacle_clipped=False),
        timings_ms={"phase2": 30.0, "phase3": 1.0})
    s = seen(res, MEASURED_ESTIMATION)
    assert (s["obstacle"], s["obstacle_cm"], s["obstacle_clipped"]) == (True, 18.5, False)
    assert (s["obstacle_voted"], s["obstacle_voted_cm"], s["obstacle_voted_clipped"]) == (True, 18.0, False)
    res.chain.phase2.obstacle_results = [SimpleNamespace(detected=False, distance_cm=None, clipped=False)]
    assert seen(res, MEASURED_ESTIMATION)["obstacle"] is False


@pytest.mark.software
@pytest.mark.parametrize("f, brakes", [
    (frame(voted=True, voted_cm=OBSTACLE_STOP_CM), True), (frame(voted=True, voted_cm=OBSTACLE_STOP_CM + 0.1), False),
    (frame(voted=True, voted_clipped=True), True), (frame(voted=True), False), (frame(voted_cm=5.0), False)])
def test_the_brake_decision_is_the_rules(f, brakes):
    assert oc.would_brake(f) is brakes


@pytest.mark.software
@pytest.mark.parametrize("frames, gap, present, result", [
    (at(20.0), 20.0, True, oc.OK),
    (at(23.0), 20.0, True, oc.OK),                     # 3 cm: at the tolerance
    (at(23.5), 20.0, True, oc.OFF),
    (clear(), 20.0, True, oc.MISSED),
    (at(20.0, 19) + [frame(voted=True, voted_cm=20.0)] * 21, 20.0, True, oc.MISSED),   # voted, but seen 47.5%
    (at(20.0)[:-1] + [frame()], 20.0, True, oc.MISSED),                                # the last vote is what counts
    ([frame(seen_=True, clipped=True, voted=True, voted_clipped=True)] * 40, 8.0, True, oc.OK),  # below the view
    (clear(), None, False, oc.OK),
    (clear(30) + [frame(30.0, True)] * 10, None, False, oc.OK),                        # seen, never voted
    (at(25.0), None, False, oc.WRONG),
    ([], 20.0, True, oc.MISSED),
])
def test_looks(frames, gap, present, result):
    assert oc.read_look(frames, gap, present)["result"] == result


@pytest.mark.software
def test_a_looks_numbers():
    frames = [frame(c, True, voted=True, voted_cm=c) for c in (19.0, 21.0, 22.0)] * 10
    frames += [frame(None, True, clipped=True, voted=True, voted_cm=22.0)] * 5 + clear(5)
    got = oc.read_look(frames, 20.0, True)
    assert got == {"frames": 40, "seen_pct": 87.5, "voted": False, "reported_cm": 21.0, "error_cm": 1.0,
                   "clipped_pct": 12.5, "would_brake": False, "result": oc.MISSED}
    assert oc.read_look(clear(), None, False)["reported_cm"] is None


def go(eyes, person, tmp_path, **options):
    return h.run_routine(oc.ObstacleCheck(eyes), person.console(), tmp_path / "out", options=options,
                         conditions_fn=no_conditions)


def results(tmp_path):
    return json.loads((tmp_path / "out" / "results.json").read_text())


def good_looks():
    return [[frame(None, True, clipped=True, voted=True, voted_clipped=True)] * 40] + [at(g) for g in (15, 20, 25, 30)] \
        + [clear()] * 3


@pytest.mark.software
def test_every_gap_then_the_clear_lane_pass(tmp_path):
    eyes = FakeEyes(good_looks())
    person = Person(*[""] * (8 * 2))                # per trial: Enter, keep
    res = go(eyes, person, tmp_path, object="can")
    assert res["verdict"] == h.PASS, res["criteria"]
    rows = results(tmp_path)["rows"]
    assert [(r["gap_cm"], r["state"]) for r in rows] == [(10, oc.PRESENT), (15, oc.PRESENT), (20, oc.PRESENT),
                                                        (25, oc.PRESENT), (30, oc.PRESENT)] + [(None, oc.CLEAR)] * 3
    assert [r["would_brake"] for r in rows] == [True, False, False, False, False, False, False, False]
    assert eyes.asked == [oc.FRAMES] * 8 and eyes.opened and eyes.closed
    prompts = "\n".join(person.prompts)
    assert prompts.count("from the front edge") == 5 and prompts.count("Lane CLEAR") == 3
    assert (tmp_path / "out" / "attempt_01_10cm.jpg").exists() and (tmp_path / "out" / "attempt_08_clear.jpg").exists()
    assert results(tmp_path)["state"]["range_cm"] == 30 and "measured up to 30 cm" in person.said()
    assert res["options"]["object"] == "can"


@pytest.mark.software
def test_the_plan_follows_the_settings(tmp_path):
    assert oc.ObstacleCheck().plan({"gaps": "12,18", "clear": 1}) == 3 == ROUTINES["obstacle-check"]().plan(
        {"gaps": "12;18", "clear": "1"})
    eyes = FakeEyes([at(18.0), at(12.0), clear()])
    res = go(eyes, Person(*[""] * 6), tmp_path, gaps="18,12", clear=1)
    assert [r["gap_cm"] for r in results(tmp_path)["rows"]] == [12, 18, None]        # nearest first
    assert res["trials_planned"] == 3


def rows_(*specs):
    out = []
    for gap, result, err, brake in specs:
        out.append({"gap_cm": gap, "state": oc.CLEAR if gap is None else oc.PRESENT, "result": result,
                    "error_cm": err, "would_brake": brake})
    return out


@pytest.mark.software
def test_each_criterion_on_its_own():
    r = oc.ObstacleCheck()
    good = [(10, oc.OK, None, True), (20, oc.OK, 1.0, False), (None, oc.OK, None, False)]
    assert all(c["passed"] for c in r.judge(rows_(*good)))

    def failed(*specs):
        return [c["name"] for c in r.judge(rows_(*specs)) if not c["passed"]]
    assert failed((20, oc.MISSED, None, False), (None, oc.OK, None, False)) == ["objects within 25 cm not seen",
                                                                               "worst distance error within 25 cm"]
    assert failed((20, oc.OFF, -3.5, False)) == ["worst distance error within 25 cm"]
    assert failed((20, oc.OK, 1.0, False), (30, oc.MISSED, None, False)) == []        # past need_cm: only the range
    assert failed((20, oc.OK, 1.0, False), (None, oc.WRONG, None, True)) == [
        "false alarms on a clear lane", f"brake decisions wrong (gaps over 2 cm from {OBSTACLE_STOP_CM:g} cm)"]
    assert failed((20, oc.OK, 1.0, True)) == [f"brake decisions wrong (gaps over 2 cm from {OBSTACLE_STOP_CM:g} cm)"]
    assert failed((20, oc.OK, 1.0, False), (OBSTACLE_STOP_CM + 1, oc.OK, 0.5, False)) == []   # too near to judge


@pytest.mark.software
def test_the_seen_width_is_the_widest_offset_every_object_read_ok_out_to():
    ok, miss = oc.OK, oc.MISSED
    res = [(15, -10, miss), (15, -5, ok), (15, 0, ok), (15, 5, ok), (15, 10, ok),
           (25, -5, ok), (25, 0, miss), (25, 5, ok)]
    assert oc.seen_width(res) == {15: 5, 25: None}                     # a miss at -10 caps 15 cm at 5


@pytest.mark.software
def test_every_gap_at_every_offset_and_only_the_path_is_judged(tmp_path):
    # 2 gaps x offsets -8, 0, 8; missed 8 cm right at 20 cm, outside the 4 cm path: mapped, not a fault
    looks = [at(15.0)] * 3 + [at(20.0), at(20.0), clear()] + [clear()]
    eyes = FakeEyes(looks)
    person = Person(*[""] * 14)
    res = go(eyes, person, tmp_path, gaps="15,20", laterals="-8,0,8", in_path_cm=4, clear=1)
    assert res["trials_planned"] == 7 and res["verdict"] == h.PASS, res["criteria"]
    rows = results(tmp_path)["rows"]
    assert [(r["gap_cm"], r["lateral_cm"]) for r in rows[:6]] == [(15, -8), (15, 0), (15, 8), (20, -8), (20, 0), (20, 8)]
    assert rows[5]["result"] == oc.MISSED and rows[6]["state"] == oc.CLEAR
    prompts = "\n".join(person.prompts)
    assert "its center 8 cm left of the lane's middle" in prompts and "in the middle of the lane" in prompts
    assert (tmp_path / "out" / "attempt_06_20cm_8cm.jpg").exists() and (tmp_path / "out" / "attempt_02_15cm.jpg").exists()
    state = results(tmp_path)["state"]
    assert state["range_cm"] == 20 and state["seen_width_cm"] == {"15": 8, "20": 0}
    assert "at 20 cm: seen out to 0 cm off the middle, both sides" in person.said()


@pytest.mark.software
def test_an_object_off_the_middle_counts_once_it_is_inside_the_path():
    r = oc.ObstacleCheck()
    r._in_path = 5.0
    rows = [{"gap_cm": 20, "lateral_cm": 8, "state": oc.PRESENT, "result": oc.MISSED, "error_cm": None,
             "would_brake": False},
            {"gap_cm": 20, "lateral_cm": 4, "state": oc.PRESENT, "result": oc.MISSED, "error_cm": None,
             "would_brake": False}]
    assert [c["name"] for c in r.judge(rows[:1]) if not c["passed"]] == ["worst distance error within 25 cm"]
    assert "objects within 25 cm not seen" in [c["name"] for c in r.judge(rows) if not c["passed"]]
    assert oc.ObstacleCheck().plan({"gaps": "10,20", "laterals": "-5,0,5", "clear": 2}) == 8
