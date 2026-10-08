"""
test_routine_lane_offset.py  --  src/routines/lane_offset.py

A look's offset over the frames on vision only, its spread, the share on
vision and the lane's median width; offsets in cm as estimation makes
them; the routine on fake eyes: the scale from the centered look, each
position's reported cm against the ruler, the tester's measured offset
asked, a frame saved per attempt, a redo of the centered look
re-measuring the scale, the scale said for config.py, no width meaning
no cm (and a fail), positions from the settings with the first at 0, the
camera closed; P3's 2 cm at its limit and past it.

--software  A scripted tester and fake eyes.
"""
import json

import numpy as np
import pytest

import src.routines.harness as h
import src.routines.lane_offset as lo
from src.routines import ROUTINES
from src.tests.test_routines import Person, no_conditions

ROI = 432               # the fake lane ROI's width
WIDTH = 280             # the fake lane's width in px: 14 cm -> 0.05 cm/px
SCALE = 14.0 / WIDTH


def norm(cm):
    """The lane_offset that reads cm at the fake scale."""
    return cm / (ROI / 2.0 * SCALE)


def frames(cm, n=100, status="vision", width=WIDTH, noise=0.0):
    return [{"lane_offset": norm(cm) + (noise if k % 2 else -noise), "lane_status": status, "lane_width_px": width}
            for k in range(n)]


class FakeEyes:
    def __init__(self, looks):
        self.looks, self.asked, self.closed = list(looks), [], False

    def open(self):
        return self

    def look(self, n):
        self.asked.append(n)
        f = self.looks.pop(0) if self.looks else []
        return f, (np.zeros((270, 480, 3), np.uint8) if f else None)

    def roi_width_px(self, frame):
        return ROI

    def close(self):
        self.closed = True


def go(looks, answers, tmp_path, **options):
    eyes = FakeEyes(looks)
    person = Person(*answers)
    res = h.run_routine(lo.LaneOffset(eyes), person.console(), tmp_path / "out", options=options,
                        conditions_fn=no_conditions)
    return res, person, eyes


def rows(tmp_path):
    return json.loads((tmp_path / "out" / "results.json").read_text())["rows"]


def answers(*measured):
    out = []
    for m in measured:
        out += ["", str(m), ""]         # placed, the ruler, keep
    return out


EXACT = [frames(0, noise=0.01), frames(2), frames(-2), frames(4), frames(-4)]


# =============================================================================
# The numbers
# =============================================================================

@pytest.mark.software
def test_a_looks_offset_is_over_frames_on_vision():
    f = frames(1, n=6) + frames(-5, n=2, status="hold") + frames(1, n=2, status="stale", width=None)
    s = lo.offset_look(f)
    assert s["frames"] == 10 and s["vision_pct"] == 60.0 and s["offset"] == pytest.approx(norm(1))
    assert s["offset_sd"] == pytest.approx(0.0) and s["lane_width_px"] == WIDTH
    pair = [{"lane_offset": v, "lane_status": "vision", "lane_width_px": None} for v in (0.1, -0.1)]
    assert lo.offset_look(pair)["offset_sd"] == pytest.approx(2 ** 0.5 / 10)     # the sample sd, not 0.1
    assert lo.offset_look([]) == {"frames": 0, "vision_pct": 0.0, "offset": None, "offset_sd": None,
                                  "lane_width_px": None}
    one = lo.offset_look(frames(1, n=1, width=None))
    assert one["offset_sd"] is None and one["lane_width_px"] is None


@pytest.mark.software
def test_the_lane_width_is_the_median():
    f = [{"lane_offset": 0.0, "lane_status": "vision", "lane_width_px": w} for w in (270, 280, 400)]
    assert lo.offset_look(f)["lane_width_px"] == 280


@pytest.mark.software
def test_to_cm_as_estimation_does():
    assert lo.to_cm(0.5, 400, 0.05) == pytest.approx(5.0)
    assert lo.to_cm(None, 400, 0.05) is None and lo.to_cm(0.5, None, 0.05) is None and lo.to_cm(0.5, 400, None) is None


# =============================================================================
# The routine
# =============================================================================

@pytest.mark.software
def test_five_positions_against_the_ruler_pass(tmp_path):
    res, person, eyes = go(EXACT, answers(0, 2, -2, 4, -4), tmp_path)
    assert res["verdict"] == h.PASS, res["criteria"]
    r = rows(tmp_path)
    assert [x["planned_cm"] for x in r] == [0, 2, -2, 4, -4]
    assert [x["reported_cm"] for x in r] == [0.0, 2.0, -2.0, 4.0, -4.0]
    assert r[0]["sd_cm"] == pytest.approx(lo.to_cm(0.01, ROI, SCALE) * (100 / 99) ** 0.5, abs=0.01)
    assert r[1]["error_cm"] == 0.0 and r[0]["lane_width_px"] == WIDTH and r[1]["vision_pct"] == 100.0
    assert eyes.asked == [lo.FRAMES] * 5 and eyes.closed
    state = json.loads((tmp_path / "out" / "results.json").read_text())["state"]
    assert state["cm_per_px"] == pytest.approx(SCALE)
    assert "set cm_per_px=0.05000 in MEASURED_ESTIMATION" in person.said() and "14 cm = 280 px" in person.said()
    prompts = "\n".join(person.prompts)
    assert "centered in the lane" in prompts
    assert prompts.index("2 cm RIGHT") < prompts.index("2 cm LEFT") < prompts.index("4 cm RIGHT") < prompts.index("4 cm LEFT")
    assert (tmp_path / "out" / "attempt_01_+0cm.jpg").exists() and (tmp_path / "out" / "attempt_03_-2cm.jpg").exists()


@pytest.mark.software
def test_the_error_is_against_what_the_tester_measured(tmp_path):
    res, _, _ = go(EXACT, answers(0, 2.5, -2, 4, -4), tmp_path)
    assert rows(tmp_path)[1]["measured_cm"] == 2.5 and rows(tmp_path)[1]["error_cm"] == -0.5
    assert res["verdict"] == h.PASS


@pytest.mark.software
def test_the_scale_comes_from_the_centered_look_only(tmp_path):
    looks = [frames(0, width=350)] + [frames(c * 350 / WIDTH, width=200) for c in (2, -2, 4, -4)]
    go(looks, answers(0, 2, -2, 4, -4), tmp_path)
    assert [x["reported_cm"] for x in rows(tmp_path)] == [0.0, 2.0, -2.0, 4.0, -4.0]   # at 14/350 cm/px


@pytest.mark.software
def test_a_redo_of_the_centered_look_measures_the_scale_again(tmp_path):
    looks = [frames(0, width=350), frames(0)] + EXACT[1:]
    a = ["", "0", "r", "", "0", ""] + answers(2, -2, 4, -4)
    res, _, _ = go(looks, a, tmp_path)
    assert res["verdict"] == h.PASS
    assert json.loads((tmp_path / "out" / "results.json").read_text())["state"]["cm_per_px"] == pytest.approx(SCALE)


@pytest.mark.software
def test_no_lane_width_no_scale_no_cm_and_a_fail(tmp_path):
    looks = [frames(0, width=None)] + EXACT[1:]
    res, person, _ = go(looks, answers(0, 2, -2, 4, -4), tmp_path)
    assert res["verdict"] == h.FAIL and res["criteria"][0]["value"] is None
    assert rows(tmp_path)[1]["reported_cm"] is None and "no lane width measured" in person.said()
    assert "ground scale" not in person.said()


@pytest.mark.software
def test_a_position_never_on_vision_fails(tmp_path):
    looks = EXACT[:2] + [frames(-2, status="stale")] + EXACT[3:]
    res, _, _ = go(looks, answers(0, 2, -2, 4, -4), tmp_path)
    assert res["verdict"] == h.FAIL and rows(tmp_path)[2]["vision_pct"] == 0.0


@pytest.mark.software
def test_a_look_with_no_frames_has_no_reading(tmp_path):
    res, _, _ = go(EXACT[:1], answers(0, 2), tmp_path, positions="0,2")
    assert res["verdict"] == h.FAIL and rows(tmp_path)[1]["reported_cm"] is None


@pytest.mark.software
def test_positions_and_frames_from_the_settings(tmp_path):
    res, _, eyes = go([frames(0), frames(3)], answers(0, 3), tmp_path, positions="0,3", frames=50, lane_cm=14)
    assert res["trials_planned"] == 2 and res["verdict"] == h.PASS and eyes.asked == [50, 50]
    with pytest.raises(ValueError, match="first position must be 0"):
        lo.LaneOffset().plan({"positions": "2,0"})


@pytest.mark.software
def test_lane_cm_sets_the_scale(tmp_path):
    go([frames(0), frames(2)], answers(0, 2), tmp_path, positions="0,2", lane_cm=28)
    assert rows(tmp_path)[1]["reported_cm"] == 4.0


@pytest.mark.software
def test_registered():
    assert ROUTINES["lane-offset"] is lo.LaneOffset and lo.LaneOffset.trials == 5 and lo.FRAMES == 100


@pytest.mark.software
def test_stopping_at_the_first_prompt_closes_the_camera(tmp_path):
    res, person, eyes = go([], ["q"], tmp_path)
    assert res["stopped_early"] and eyes.closed and "ground scale" not in person.said()


# =============================================================================
# The criterion
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("off, verdict", [(2.0, h.PASS), (2.01, h.FAIL), (-2.01, h.FAIL), (-2.0, h.PASS)])
def test_p3s_two_cm(tmp_path, off, verdict):
    looks = EXACT[:3] + [frames(4 + off)] + EXACT[4:]
    res, _, _ = go(looks, answers(0, 2, -2, 4, -4), tmp_path)
    assert res["verdict"] == verdict
    assert res["criteria"][0]["value"] == pytest.approx(abs(off))


@pytest.mark.software
def test_the_real_eyes_lane_roi_width():
    from src.routines.camera_look import Eyes
    eyes = Eyes(source=object()).open()
    assert eyes.roi_width_px(np.zeros((270, 480, 3), np.uint8)) == 432       # course.md: 5-95% across
    assert eyes.roi_width_px(np.zeros((360, 640, 3), np.uint8)) == 576
    from types import SimpleNamespace
    from src.perception.roi_crop import ROIBounds
    half = Eyes(source=object(), config=SimpleNamespace(roi=SimpleNamespace(lane=ROIBounds(0.25, 0.7, 0.75, 1.0))))
    assert half.roi_width_px(np.zeros((270, 480, 3), np.uint8)) == 240         # read from the config
