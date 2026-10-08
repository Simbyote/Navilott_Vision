"""
test_routine_detect_range.py  --  src/routines/detect_range.py, src/routines/camera_look.py

camera_look: a frame's result as a Seen dict (lights and signs at their
confidence gates), looks through the real chain on black frames (drops
skipped, the source ending, a fresh processor each look), the source
closed, frames saved, --set lists parsed. detect_range: a look read OK,
MISSED or WRONG for every light and sign state; the reliable range; the
routine on fake eyes: gaps times states planned from the settings, the
robot moved once per gap, every state prompted, a JPEG per attempt, a
redo repeating its state, the range in the summary, a bad target refused;
the criteria one at a time and at need_cm.

--software  A scripted tester and fake eyes; black frames through the real chain.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import src.routines.detect_range as dr
import src.routines.harness as h
from src.config import MEASURED_ESTIMATION
from src.params import GREEN, RED, STOP_SIGN, TRAFFIC_LIGHT, YELLOW
from src.routines import ROUTINES
from src.routines.camera_look import Eyes, parse_numbers, save_frame, seen
from src.tests.test_routines import Person, no_conditions


def frame(light=None, sign=False, drive="go", stop_sign=False):
    return {"light": light, "sign": sign, "drive_state": drive, "stop_sign": stop_sign}


def look(n=40, **kw):
    return [frame(**kw) for _ in range(n)]


# =============================================================================
# camera_look
# =============================================================================

def det(type_, label, conf):
    return SimpleNamespace(type=type_, label_detail=label, confidence=conf)


def result(dets):
    return SimpleNamespace(
        chain=SimpleNamespace(phase2=SimpleNamespace(detections=dets),
                              offset=SimpleNamespace(offset=0.1, mode="two", lane_width_px=300)),
        packet=SimpleNamespace(drive_state="stop", stop_sign_detected=True, lane_offset=0.08, lane_status="vision"),
        timings_ms={"phase2": 30.0, "phase3": 1.5, "capture": 9.0})


@pytest.mark.software
def test_seen_counts_lights_and_signs_at_their_gates():
    p3 = MEASURED_ESTIMATION
    s = seen(result([det(TRAFFIC_LIGHT, RED, p3.min_confidence_traffic),
                     det(STOP_SIGN, "stop", p3.min_confidence_sign)]), p3)
    assert s == {"light": RED, "sign": True, "drive_state": "stop", "stop_sign": True, "lane_offset": 0.08,
                 "lane_status": "vision", "p2_offset": 0.1, "lane_mode": "two", "lane_width_px": 300,
                 "total_ms": 31.5}
    under = seen(result([det(TRAFFIC_LIGHT, RED, p3.min_confidence_traffic - 0.01),
                         det(STOP_SIGN, "stop", p3.min_confidence_sign - 0.01)]), p3)
    assert under["light"] is None and under["sign"] is False


class Source:
    """Black frames; drop at the reads listed; ends after `frames` reads."""
    fps, label = 20, "fake"

    def __init__(self, frames=100, drops=()):
        self.reads, self.frames, self.drops, self.closed = 0, frames, set(drops), False

    def read(self):
        self.reads += 1
        if self.reads > self.frames:
            return None
        if self.reads in self.drops:
            return None, None, None
        return np.zeros((270, 480, 3), np.uint8), self.reads, self.reads * 50

    def close(self):
        self.closed = True


@pytest.mark.software
def test_a_look_runs_the_chain_skipping_drops_until_n_frames():
    src = Source(drops={2, 3})
    eyes = Eyes(src).open()
    frames, last = eyes.look(4)
    assert len(frames) == 4 and src.reads == 6 and last.shape == (270, 480, 3)
    assert frames[0]["drive_state"] == "go" and frames[0]["light"] is None
    eyes.close()
    assert src.closed


@pytest.mark.software
def test_a_look_ends_short_when_the_source_ends_or_keeps_dropping():
    assert len(Eyes(Source(frames=2)).open().look(5)[0]) == 2
    src = Source(drops=set(range(1, 100)))
    frames, last = Eyes(src).open().look(3)
    assert frames == [] and last is None and src.reads == 9          # 3 reads a frame, then it gives up


@pytest.mark.software
def test_each_look_starts_a_fresh_processor(monkeypatch):
    import src.phase3_linker as p3l
    made = []
    real = p3l.make_processor
    monkeypatch.setattr(p3l, "make_processor", lambda *a: made.append(1) or real(*a))
    eyes = Eyes(Source()).open()
    eyes.look(2)
    eyes.look(2)
    assert len(made) == 2


@pytest.mark.software
def test_save_frame_and_parse_numbers(tmp_path):
    assert save_frame(tmp_path / "a.jpg", np.zeros((10, 10, 3), np.uint8)) and (tmp_path / "a.jpg").exists()
    assert not save_frame(tmp_path / "b.jpg", None)
    assert not save_frame(tmp_path / "no_dir" / "c.jpg", np.zeros((10, 10, 3), np.uint8))
    assert parse_numbers("0, 10;20,,") == [0.0, 10.0, 20.0] and parse_numbers(30) == [30.0]
    assert parse_numbers(12.5) == [12.5]


# =============================================================================
# Reading a look
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("frames, state, result", [
    (look(light=RED, drive="stop"), RED, dr.OK),
    (look(light=YELLOW, drive="caution"), YELLOW, dr.OK),
    (look(light=GREEN), GREEN, dr.OK),
    (look(), GREEN, dr.MISSED),                                   # votes go, but never saw it
    (look(), RED, dr.MISSED),
    (look(light=YELLOW, drive="caution"), RED, dr.WRONG),
    (look(light=RED, drive="stop"), GREEN, dr.WRONG),
    (look(light=GREEN), RED, dr.WRONG),                           # votes go like nothing seen: the frames tell
    (look(n=20, light=GREEN) + look(n=20), RED, dr.WRONG),
    (look(n=19, light=GREEN) + look(n=21), RED, dr.MISSED),
    (look(n=30) + look(n=10, light=YELLOW, drive="caution"), RED, dr.WRONG),     # a few frames, but it voted
    (look(n=30, light=RED, drive="stop") + look(n=10, light=RED, drive="go"), RED, dr.MISSED),   # the last vote
    (look(), dr.OFF, dr.OK),
    (look(light=RED, drive="stop"), dr.OFF, dr.WRONG),
    (look(n=20, light=RED, drive="stop") + look(n=20, light=None, drive="stop"), RED, dr.OK),     # 50%: enough
    (look(n=19, light=RED, drive="stop") + look(n=21, light=None, drive="stop"), RED, dr.MISSED),
    ([], RED, dr.MISSED),
])
def test_light_looks(frames, state, result):
    assert dr.read_look(frames, dr.LIGHT, state)["result"] == result


@pytest.mark.software
@pytest.mark.parametrize("frames, state, result", [
    (look(sign=True, stop_sign=True), dr.PRESENT, dr.OK),
    (look(), dr.PRESENT, dr.MISSED),
    (look(), dr.ABSENT, dr.OK),
    (look(sign=True, stop_sign=True), dr.ABSENT, dr.WRONG),
    (look(n=10, sign=True, stop_sign=False) + look(n=30), dr.ABSENT, dr.OK),     # seen, not voted: counted
])
def test_sign_looks(frames, state, result):
    assert dr.read_look(frames, dr.SIGN, state)["result"] == result


@pytest.mark.software
def test_a_looks_numbers():
    frames = look(n=30, light=RED, drive="stop") + look(n=6, light=YELLOW, drive="stop") + look(n=4, drive="stop")
    assert dr.read_look(frames, dr.LIGHT, RED) == {"frames": 40, "seen_pct": 75.0, "wrong_frames": 6,
                                                    "voted": "stop", "expected": "stop", "result": dr.OK}
    off = dr.read_look(look(n=8, light=GREEN) + look(n=32), dr.LIGHT, dr.OFF)
    assert (off["seen_pct"], off["wrong_frames"], off["result"]) == (20.0, 8, dr.OK)
    sign = dr.read_look(look(n=10, sign=True, stop_sign=True) + look(n=30, stop_sign=True), dr.SIGN, dr.PRESENT)
    assert (sign["seen_pct"], sign["wrong_frames"], sign["result"]) == (25.0, 0, dr.MISSED)
    gone = dr.read_look(look(n=10, sign=True) + look(n=30), dr.SIGN, dr.ABSENT)
    assert (gone["seen_pct"], gone["wrong_frames"], gone["result"]) == (25.0, 10, dr.OK)


@pytest.mark.software
def test_the_reliable_range():
    ok, miss = dr.OK, dr.MISSED
    assert dr.reliable_range([(0, ok), (10, ok), (20, ok), (30, miss), (45, ok)]) == 20
    assert dr.reliable_range([(0, ok), (10, ok)]) == 10
    assert dr.reliable_range([(0, miss), (10, ok)]) is None
    assert dr.reliable_range([(10, ok), (0, ok), (10, miss)]) == 0


# =============================================================================
# The routine
# =============================================================================

class FakeEyes:
    def __init__(self, looks):
        self.looks, self.asked, self.opened, self.closed = list(looks), [], False, False

    def open(self):
        self.opened = True
        return self

    def look(self, n):
        self.asked.append(n)
        return (self.looks.pop(0) if self.looks else []), np.zeros((8, 8, 3), np.uint8)

    def close(self):
        self.closed = True


LIGHT_OK = [look(light=RED, drive="stop"), look(light=YELLOW, drive="caution"), look(light=GREEN), look()]
SIGN_OK = [look(sign=True, stop_sign=True), look()]


def go(eyes, person, tmp_path, **options):
    return h.run_routine(dr.DetectRange(eyes), person.console(), tmp_path / "out", options=options,
                         conditions_fn=no_conditions)


def results(tmp_path):
    return json.loads((tmp_path / "out" / "results.json").read_text())


@pytest.mark.software
def test_every_state_at_every_gap_pass(tmp_path):
    eyes = FakeEyes(LIGHT_OK * 5)
    person = Person(*[""] * (5 * (1 + 4 + 4)))      # per gap: the move, then per state: set + keep
    res = go(eyes, person, tmp_path)
    assert res["verdict"] == h.PASS, res["criteria"]
    rows = results(tmp_path)["rows"]
    assert [(r["gap_cm"], r["state"]) for r in rows[:5]] == [(0, RED), (0, YELLOW), (0, GREEN), (0, dr.OFF), (10, RED)]
    assert len(rows) == 20 and eyes.asked == [dr.FRAMES] * 20 and eyes.opened and eyes.closed
    prompts = "\n".join(person.prompts)
    assert prompts.count("cm before the stop line") == 5 and "light YELLOW" in prompts and "light OFF" in prompts
    assert (tmp_path / "out" / "attempt_01_0cm_red.jpg").exists()
    assert (tmp_path / "out" / "attempt_20_45cm_off.jpg").exists()
    assert results(tmp_path)["state"]["range_cm"] == 45
    assert "every state read right up to 45 cm" in person.said()


@pytest.mark.software
def test_the_sign_with_its_own_gaps_and_frames(tmp_path):
    eyes = FakeEyes(SIGN_OK * 2)
    res = go(eyes, Person(*[""] * 2 * (1 + 2 + 2)), tmp_path, target="sign", gaps="5,15", frames=10, need_cm=15)
    assert res["verdict"] == h.PASS and res["trials_planned"] == 4
    assert [(r["gap_cm"], r["state"]) for r in results(tmp_path)["rows"]] == \
        [(5, dr.PRESENT), (5, dr.ABSENT), (15, dr.PRESENT), (15, dr.ABSENT)]
    assert eyes.asked == [10] * 4


@pytest.mark.software
def test_one_gap_as_a_number(tmp_path):
    res = go(FakeEyes(SIGN_OK), Person(*[""] * 5), tmp_path, target="sign", gaps=10)
    assert res["trials_planned"] == 2 and res["verdict"] == h.PASS


@pytest.mark.software
def test_a_redo_looks_again_at_the_same_state(tmp_path):
    eyes = FakeEyes([look(), look(light=RED, drive="stop")] + LIGHT_OK[1:] + LIGHT_OK * 4)
    answers = ["", "", "r", "", ""] + [""] * (3 * 2 + 4 * 9)
    person = Person(*answers)
    res = go(eyes, person, tmp_path)
    assert res["verdict"] == h.PASS, res["criteria"]
    assert results(tmp_path)["rows"][0]["state"] == RED and "MISSED: voted go, expected stop" in person.said()
    assert (tmp_path / "out" / "attempt_01_0cm_red.jpg").exists() and (tmp_path / "out" / "attempt_02_0cm_red.jpg").exists()


@pytest.mark.software
def test_a_bad_target_is_refused():
    with pytest.raises(ValueError, match="light or sign"):
        dr.DetectRange().plan({"target": "cone"})


@pytest.mark.software
def test_registered_and_planned_from_the_settings():
    assert ROUTINES["detect-range"] is dr.DetectRange
    assert dr.FRAMES == 40 and dr.DetectRange.trials == 20 and dr.DetectRange().plan({"gaps": "0,10", "target": "sign"}) == 4


@pytest.mark.software
def test_stopping_at_the_first_prompt_closes_the_camera(tmp_path):
    eyes = FakeEyes([])
    person = Person("q")
    res = go(eyes, person, tmp_path)
    assert res["stopped_early"] and eyes.closed and "\nrange: " not in person.said()


@pytest.mark.software
def test_no_range_when_the_nearest_gap_fails(tmp_path):
    person = Person(*[""] * 5)
    go(FakeEyes([look(), look()]), person, tmp_path, target="sign", gaps=0)
    assert "not even at the nearest gap" in person.said() and results(tmp_path)["state"]["range_cm"] is None


# =============================================================================
# The criteria
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("looks, failing", [
    (LIGHT_OK * 2 + [look()] + LIGHT_OK[1:] + LIGHT_OK * 2,
     ["trials not read right within 20 cm"]),                                 # red missed at 20 cm
    (LIGHT_OK * 3 + [look()] + LIGHT_OK[1:] + LIGHT_OK, []),                  # missed at 30 cm: only the range
    (LIGHT_OK * 4 + [look(light=GREEN)] + LIGHT_OK[1:],
     ["wrong reads at any gap (something that isn't there)"]),                # red read as green at 45 cm
    (LIGHT_OK * 4 + LIGHT_OK[:3] + [look(light=RED, drive="stop")],
     ["wrong reads at any gap (something that isn't there)"]),                # off read as red
    ([LIGHT_OK[0], look(light=RED, drive="stop")] + LIGHT_OK[2:] + LIGHT_OK * 4,
     ["trials not read right within 20 cm", "wrong reads at any gap (something that isn't there)"]),
])
def test_each_criterion(tmp_path, looks, failing):
    res = go(FakeEyes(looks), Person(*[""] * 45), tmp_path)
    assert [c["name"] for c in res["criteria"] if not c["passed"]] == failing


@pytest.mark.software
def test_need_cm_moves_the_line(tmp_path):
    looks = LIGHT_OK * 3 + [look()] + LIGHT_OK[1:] + LIGHT_OK          # red missed at 30 cm
    res = go(FakeEyes(looks), Person(*[""] * 45), tmp_path, need_cm=30)
    assert [c["name"] for c in res["criteria"] if not c["passed"]] == ["trials not read right within 30 cm"]
