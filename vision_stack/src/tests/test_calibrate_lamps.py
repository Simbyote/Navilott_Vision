"""
test_calibrate_lamps.py  --  src/scripts/calibrate_lamps.py

Synthetic frames hold one lamp each, drawn in HSV with a known core, ring
(the lamp's color), glow and background, at the robot's frame size and in
its traffic ROI; SCENE_CONFIG runs them through preprocess without
undistortion. Known answers: the band spans the lamp's hue and puts S and
V between the lamp and its glow; red across the wrap splits into both
halves; the blob area under the band is the lamp's, not its glow's; the
warnings (no separation, shared hues); the file written keeps what wasn't
measured and loads.

--software  Synthetic frames in a temp folder. No camera.
"""
import json
import math

import cv2
import numpy as np
import pytest

import src.scripts.calibrate_lamps as cl
from src.params import FRAME_H, FRAME_W, HSV_RANGES_PATH
from src.perception.color_branch import load_hsv_ranges
from src.tests.scenes import SCENE_CONFIG

CENTER = (216, 50)          # frame px: inside roi_crop.TRAFFIC (x 168-264, y 0-108 at 480x270)
LAMP_R, GLOW_R = 9, 24      # the drawn lamp (~250 px^2) and its glow
BACKGROUND = (85, 11, 205)  # pale and bright, as the 2026-10-04 green profile's background

# (core, ring, glow) in OpenCV HSV, after the robot's lamp profiles
GREEN = ((99, 30, 255), (100, 52, 245), (97, 28, 212))     # a cyan "green": hue ~100 on camera
YELLOW = ((24, 40, 255), (25, 140, 245), (24, 60, 205))
RED = ((179, 60, 255), (2, 160, 240), (3, 70, 205))        # across the wrap: 176-179 and 0-6


def lamp_frame(core, ring, glow, center=CENTER):
    hsv = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    hsv[:] = BACKGROUND
    cv2.circle(hsv, center, GLOW_R, glow, -1)
    cv2.circle(hsv, center, LAMP_R, ring, -1)
    cv2.circle(hsv, center, 3, core, -1)
    if ring[0] < 10:                                       # red: half the ring on the other side of the wrap
        cv2.ellipse(hsv, center, (LAMP_R, LAMP_R), 0, 0, 180, (176, ring[1], ring[2]), -1)
        cv2.circle(hsv, center, 3, core, -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def measured(color, lamp):
    roi = cl.traffic_roi(lamp_frame(*lamp), SCENE_CONFIG)
    m = cl.measure_lamp(roi)
    return m, cl.bands_for(color, m["band"]), roi


def write_frames(tmp_path, name, lamp, n=3):
    folder = tmp_path / name / "frames"
    folder.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"{i:06d}.png"), lamp_frame(*lamp))
    return tmp_path / name


# =============================================================================
# Measuring one lamp
# =============================================================================

@pytest.mark.software
def test_the_lamp_is_found_at_its_center_in_the_traffic_roi():
    m, _, _ = measured("green", GREEN)
    x0 = round(0.35 * FRAME_W)                             # TRAFFIC's left edge
    assert m["center"] == pytest.approx((CENTER[0] - x0, CENTER[1]), abs=1.0)


@pytest.mark.software
def test_the_band_spans_the_lamps_hue_and_puts_s_and_v_between_lamp_and_glow():
    m, e, _ = measured("green", GREEN)
    lo, hi = e["green"]["lower"], e["green"]["upper"]
    assert lo[0] <= 99 and hi[0] >= 100 and hi[0] - lo[0] <= 2 + 2 * cl.HUE_MARGIN + 2
    assert GREEN[2][1] < lo[1] <= GREEN[1][1], "S: above the glow's, at or under the lamp's"
    assert GREEN[2][2] < lo[2] <= GREEN[1][2], "V: above the glow's, at or under the lamp's"
    assert hi[1:] == [255, 255] and m["s_gap"] > 0 and m["v_gap"] > 0


@pytest.mark.software
def test_the_blob_under_the_band_is_the_lamp_not_its_glow():
    _, e, roi = measured("green", GREEN)
    area = cl.blob_area(roi, e)
    assert area == pytest.approx(math.pi * LAMP_R ** 2, rel=0.2)
    assert area < 0.5 * math.pi * GLOW_R ** 2


@pytest.mark.software
def test_red_across_the_wrap_gets_both_halves():
    m, e, roi = measured("red", RED)
    assert set(e) == {"red_low", "red_high"}
    assert e["red_high"]["lower"][0] <= 176 and e["red_high"]["upper"][0] == 180
    assert e["red_low"]["lower"][0] == 0 and e["red_low"]["upper"][0] >= 2
    assert cl.blob_area(roi, e) == pytest.approx(math.pi * LAMP_R ** 2, rel=0.2)


@pytest.mark.software
@pytest.mark.parametrize("lo, hi, low, high", [
    (172.0, 185.0, (0, 5), (172, 180)),     # across the wrap
    (2.0, 9.0, (2, 9), (180, 180)),         # all low: the high half is the empty hue 180
    (168.0, 178.0, (0, 0), (168, 180)),     # all high: the low half is hue 0 alone, still red
    (188.0, 198.0, (8, 18), (180, 180)),    # unwrapped but all past 180 (an orange-shifted red): the low side
])
def test_red_bands_split_or_fill_in_the_other_half(lo, hi, low, high):
    e = cl.bands_for("red", {"h_lo": lo, "h_hi": hi, "s_min": 100.0, "v_min": 200.0})
    assert (e["red_low"]["lower"][0], e["red_low"]["upper"][0]) == low
    assert (e["red_high"]["lower"][0], e["red_high"]["upper"][0]) == high


@pytest.mark.software
def test_a_lamp_its_glow_hides_is_reported():
    flat = ((99, 30, 230), (100, 30, 230), (98, 32, 232))           # glow as bright and pale as the lamp
    m, _, _ = measured("green", flat)
    assert m["s_gap"] <= 0 and m["v_gap"] <= 0


@pytest.mark.software
def test_colors_sharing_hues_are_reported():
    orange = ((15, 60, 255), (17, 150, 240), (17, 70, 205))         # an overexposed red gone orange
    per_color = {"red": cl.bands_for("red", measured("red", orange)[0]["band"]),
                 "yellow": measured("yellow", YELLOW)[1], "green": measured("green", GREEN)[1]}
    assert cl.hue_overlaps(per_color) == [("red", "yellow")]
    clean = {**per_color, "red": measured("red", RED)[1]}
    assert cl.hue_overlaps(clean) == []


# =============================================================================
# Frames, the file and the command line
# =============================================================================

@pytest.mark.software
def test_a_run_folder_is_searched_and_its_frames_spread(tmp_path):
    folder = write_frames(tmp_path, "run", GREEN, n=8)
    assert len(cl.read_frames(str(folder), limit=3)) == 3
    assert len(cl.read_frames(str(folder / "frames" / "000000.png"))) == 1
    with pytest.raises(FileNotFoundError):
        cl.read_frames(str(tmp_path / "missing"))


@pytest.mark.software
def test_the_command_line_prints_each_band_and_area_and_writes_only_with_write(tmp_path):
    out = tmp_path / "hsv.json"
    out.write_text(HSV_RANGES_PATH.read_text())
    before = json.loads(out.read_text())
    lines = []
    args = [f"green={write_frames(tmp_path, 'g', GREEN)}", f"red={write_frames(tmp_path, 'r', RED)}", f"--out={out}"]
    assert cl.main(args, config=SCENE_CONFIG, say=lines.append) == 0
    text = "\n".join(lines)
    assert "green:" in text and "red_low" in text and "blob area under it" in text and "not written" in text
    assert json.loads(out.read_text()) == before
    assert cl.main(args + ["--write"], config=SCENE_CONFIG, say=lines.append) == 0
    after = json.loads(out.read_text())
    assert after["yellow"] == before["yellow"], "a color not measured keeps its band"
    assert after["green"] != before["green"] and after["green"]["lower"][0] >= 90
    assert load_hsv_ranges(str(out)).is_calibrated


@pytest.mark.software
def test_a_bad_argument_is_exit_2_and_a_band_the_loader_refuses_is_not_written(tmp_path):
    lines = []
    assert cl.main(["blue=x.png"], config=SCENE_CONFIG, say=lines.append) == 2
    assert cl.main(["green=nowhere"], config=SCENE_CONFIG, say=lines.append) == 2
    out = tmp_path / "hsv.json"
    out.write_text(HSV_RANGES_PATH.read_text())
    before = out.read_text()
    with pytest.raises(ValueError, match="doesn't load"):
        cl.write_ranges(out, {"green": {"lower": [90, 40, 230], "upper": [80, 255, 255]}})
    assert out.read_text() == before and not list(tmp_path.glob("*.tmp.json"))
