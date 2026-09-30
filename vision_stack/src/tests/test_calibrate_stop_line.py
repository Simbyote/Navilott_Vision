"""
test_calibrate_stop_line.py  --  src/scripts/calibrate_stop_line.py

Measuring a tape mark on drawn frames through the real Phase 2, the reasons
a mark isn't recorded, the interactive loop (undo, too few marks, a typo, a
mark the detector can't see), and the file it writes, loaded back through
load_stop_line_table. No camera.
"""
import json

import numpy as np
import pytest

import src.scripts.calibrate_stop_line as cs
from src.config import MEASURED
from src.params import FRAME_H, FRAME_W
from src.perception.stop_line_table import load_stop_line_table
from src.tests.scenes import LANE_RECT, SCENE_CONFIG, scene

ROI_H = LANE_RECT[3]
THICK = 6


def tape(y_top):
    """A frame with a tape strip whose top edge is y_top lane-ROI rows down: its near edge is y_top + THICK."""
    return scene(stop_line=(120, 320, y_top), stop_line_thickness=THICK)


# =============================================================================
# Measuring a mark
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("y_top", [10, 30, 50])
def test_a_tape_mark_measures_its_rows_above_the_roi_bottom(y_top):
    m = cs.measure_mark([tape(y_top)] * 4, SCENE_CONFIG)
    assert m["seen"] == m["frames"] == 4 and m["clipped"] == 0 and m["spread"] == 0.0
    assert m["rows"] == pytest.approx(ROI_H - (y_top + THICK - 1), abs=1.5)
    assert cs.mark_problem(m) is None


@pytest.mark.software
def test_the_rows_are_the_median_over_the_frames():
    frames = [tape(30)] * 3 + [tape(10)] * 2
    m = cs.measure_mark(frames, SCENE_CONFIG)
    assert m["rows"] == cs.measure_mark([tape(30)], SCENE_CONFIG)["rows"] and m["spread"] > 0


@pytest.mark.software
def test_a_mark_seen_in_too_few_frames_is_not_recorded():
    m = cs.measure_mark([tape(30)] + [scene()] * 3, SCENE_CONFIG)
    assert m["seen"] == 1 and "seen in only 1 of 4" in cs.mark_problem(m)


@pytest.mark.software
def test_a_mark_under_the_view_bottom_is_not_recorded():
    m = cs.measure_mark([tape(ROI_H - 5)] * 3, SCENE_CONFIG)
    assert m["clipped"] == m["seen"] == 3 and "off the bottom" in cs.mark_problem(m)


@pytest.mark.software
def test_no_line_at_all_measures_nothing():
    m = cs.measure_mark([scene()] * 3, SCENE_CONFIG)
    assert (m["rows"], m["seen"], m["spread"]) == (None, 0, 0.0) and cs.mark_problem(m)


# =============================================================================
# The interactive loop
# =============================================================================

def interactive(answers, frames_for):
    """collect_marks with typed answers; frames_for(cm) -> the frame the camera sees for that mark."""
    said, typed = [], iter(answers)
    last = {"cm": None}

    def ask(prompt):
        text = next(typed)
        try:
            last["cm"] = float(text)
        except ValueError:
            pass
        return text

    marks = cs.collect_marks(lambda n: [frames_for(last["cm"])] * n, ask, said.append, SCENE_CONFIG, 3)
    return marks, said


@pytest.mark.software
def test_the_loop_records_marks_undoes_and_needs_three():
    rows_for = {3.0: 70, 6.0: 50, 10.0: 30, 15.0: 10, 99.0: 40}
    marks, said = interactive(["3", "", "6", "99", "u", "abc", "10", "15", ""],
                              lambda cm: tape(rows_for[cm]))
    assert [m["cm"] for m in marks] == [3.0, 6.0, 10.0, 15.0]
    assert [m["rows"] for m in marks] == sorted(m["rows"] for m in marks)       # farther marks sit higher
    assert any("need at least 3" in s for s in said)
    assert any("removed the 99.0 cm mark" in s for s in said)
    assert any("isn't a distance" in s for s in said)


@pytest.mark.software
def test_a_mark_the_detector_cannot_see_is_not_recorded():
    frames = {3.0: tape(70), 5.0: scene(), 6.0: tape(50), 10.0: tape(30)}
    marks, said = interactive(["3", "5", "6", "10", ""], lambda cm: frames[cm])
    assert [m["cm"] for m in marks] == [3.0, 6.0, 10.0]
    assert any("not recorded" in s for s in said)


# =============================================================================
# The file
# =============================================================================

@pytest.mark.software
def test_parse_marks():
    assert cs.parse_marks("3:4.5, 6:21") == [{"cm": 3.0, "rows": 4.5}, {"cm": 6.0, "rows": 21.0}]
    with pytest.raises(ValueError, match="cm:rows"):
        cs.parse_marks("3-4.5")


def marks_text(rows=(0.0, 15.0, 30.0, 50.0, 70.0), a=800.0, b=160.0, c=-2.0, noise=None):
    cm = a / (b - np.array(rows)) + c + (0 if noise is None else np.array(noise))
    return ",".join(f"{y:.4f}:{r}" for y, r in zip(cm, rows))


@pytest.mark.software
def test_a_refit_writes_a_file_the_pipeline_loads_back(tmp_path):
    out, said = tmp_path / "t.json", []
    assert cs.main(["--marks", marks_text(), "--out", str(out), "--reference", "the bumper"], say=said.append) == 0
    rec = json.loads(out.read_text())
    assert rec["curve"] == pytest.approx([800.0, 160.0, -2.0], rel=1e-3) and rec["reference"] == "the bumper"
    assert len(rec["marks"]) == 5 and rec["error_cm"]["max"] < 0.01
    t = load_stop_line_table(out, MEASURED.preprocess, (FRAME_H, FRAME_W))
    assert t is not None and t.to_cm(30.0) == pytest.approx(800 / 130 - 2, abs=1e-3) and t.max_rows == 70.0
    assert any("GOOD" in s for s in said)


@pytest.mark.software
@pytest.mark.parametrize("marks, code, words", [
    ("3:0,6:20,5:40", 1, "farther away"),
    ("3:0,6:20", 1, "at least 3"),
    ("3-0", 2, "cm:rows"),
])
def test_bad_marks_are_reported_and_nothing_is_written(tmp_path, marks, code, words):
    out, said = tmp_path / "t.json", []
    assert cs.main(["--marks", marks, "--out", str(out)], say=said.append) == code
    assert not out.exists() and any(words in s for s in said)


@pytest.mark.software
def test_the_report_gives_each_marks_error_and_the_verdict():
    rec = cs.build_record(cs.parse_marks(marks_text(noise=[0.0, 0.3, -0.2, 0.4, 0.0])), MEASURED.preprocess,
                          (FRAME_W, FRAME_H), "x")
    lines = cs.report_lines(rec, ROI_H)
    assert len([l for l in lines if l.strip()[:1].isdigit()]) == 5
    worst = rec["error_cm"]["max"]
    verdict = "GOOD" if worst < cs.GOOD_CM else "OK" if worst < cs.OK_CM else "POOR"
    assert worst > 0.05 and any(f"-> {verdict}" in l for l in lines)


@pytest.mark.software
def test_three_marks_and_short_reach_are_warned_about():
    rec = cs.build_record(cs.parse_marks(marks_text(rows=(0.0, 10.0, 20.0))), MEASURED.preprocess,
                          (FRAME_W, FRAME_H), "x")
    lines = cs.report_lines(rec, ROI_H)
    assert any("nothing checks them" in l for l in lines) and any("extrapolated" in l for l in lines)
    assert any("view bottom reads 3.0 cm" in l for l in lines)


@pytest.mark.software
def test_no_lens_calibration_stops_before_anything(monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(cs, "MEASURED", replace(MEASURED, preprocess=replace(MEASURED.preprocess,
                                                                               calibration_path=None)))
    said = []
    assert cs.main(["--marks", marks_text()], say=said.append) == 2 and "lens calibration" in said[0]


@pytest.mark.software
def test_marks_are_measured_in_rows_only_whatever_cm_calibration_exists(monkeypatch):
    from dataclasses import replace
    from src.tests.scenes import SYNTHETIC_GROUND, SYNTHETIC_STOP_LINE_TABLE
    monkeypatch.setattr(cs, "MEASURED", replace(MEASURED, ground=SYNTHETIC_GROUND,
                                                stop_line_table=SYNTHETIC_STOP_LINE_TABLE))
    seen = {}

    def collect(grab, ask, say, config, n):
        seen["config"] = config
        return cs.parse_marks(marks_text())
    monkeypatch.setattr(cs, "collect_marks", collect)
    assert cs.main(["--out", "/dev/null"], grab=lambda n: [], say=lambda s: None) == 0
    assert seen["config"].ground is None and seen["config"].stop_line_table is None
