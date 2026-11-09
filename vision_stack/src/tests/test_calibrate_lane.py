"""
test_calibrate_lane.py  --  src/scripts/calibrate_lane.py

Synthetic lanes through the real chain: a camera off the centerline sees a
centered robot's lane middle 20 px right of the ROI's, and a lane 140 px wide.
The calibration finds both, says what the current settings read, warns that a
min_lane_width_px wider than the lane throws every pair away, accounts for a
robot parked off center on purpose, and with robot_x_px set the same frames
read centered.

--software  Synthetic frames (SCENE_CONFIG). No camera.
"""
from dataclasses import replace

import pytest

from src.phase2_linker import run_chain
from src.scripts import calibrate_lane as cl
from src.tests.scenes import ROI_W, SCENE_CONFIG, synthetic_frame

pytestmark = pytest.mark.software

MIDDLE = ROI_W / 2.0
SHIFTED = [(synthetic_frame((MIDDLE + 20 - 70, MIDDLE + 20 + 70)), i, 50 * i) for i in range(6)]


def test_the_lane_middle_and_width_and_what_the_settings_read_now():
    cfg = replace(SCENE_CONFIG, lane_offset=replace(SCENE_CONFIG.lane_offset, min_lane_width_px=300.0))
    m = cl.measure(SHIFTED, cfg, run_chain)
    rec = cl.recommend(m, ROI_W)
    assert rec["robot_x_px"] == pytest.approx(MIDDLE + 20, abs=1.0) and rec["shift_px"] == pytest.approx(20, abs=1.0)
    assert rec["lane_width_px"] == pytest.approx(140, abs=1.0)
    assert rec["expected_half_lane_px"] == pytest.approx(70, abs=0.5)
    assert rec["min_lane_width_px"] == round(0.84 * rec["lane_width_px"])
    text = "\n".join(cl.report(rec, m, cfg))
    assert "robot_x_px = " in text and "wider than the lane" in text, "300 px throws every 140 px pair away"


def test_a_robot_parked_off_center_and_the_fixed_reading():
    m = cl.measure(SHIFTED, SCENE_CONFIG, run_chain)
    rec = cl.recommend(m, ROI_W, offset_cm=2.0, lane_cm=14.0)
    assert rec["robot_x_px"] == pytest.approx(MIDDLE + 20 + 20, abs=1.0), "2 cm of a 140 px, 14 cm lane is 20 px"
    assert rec["offset_now"] == pytest.approx(-20 / MIDDLE, abs=0.01), "read against the ROI's middle: left of it"
    fixed = replace(SCENE_CONFIG, lane_offset=replace(SCENE_CONFIG.lane_offset, robot_x_px=MIDDLE + 20))
    assert all(run_chain(f, i, t, fixed).offset.offset == pytest.approx(0, abs=0.01) for f, i, t in SHIFTED)


def test_no_pair_is_an_error_and_main_refuses_an_empty_source(tmp_path, capsys):
    one = [(synthetic_frame((MIDDLE - 70,)), 0, 0)]
    with pytest.raises(ValueError, match="both lane lines"):
        cl.recommend(cl.measure(one, SCENE_CONFIG, run_chain), ROI_W)
    (tmp_path / "f").mkdir()
    assert cl.main(["--frames", str(tmp_path / "f")]) == 2
    assert "lane calibration: no images" in capsys.readouterr().out
