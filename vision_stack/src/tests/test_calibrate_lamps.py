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
from src.tests.scenes import SCENE_CONFIG, SCENE_TRAFFIC_ROI as TRAFFIC

CENTER = (216, 50)          # frame px: inside SCENE_CONFIG's traffic ROI (x 144-288, y 0-94 at 480x270)
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
    x0, y0 = round(TRAFFIC.x0 * FRAME_W), round(TRAFFIC.y0 * FRAME_H)      # TRAFFIC's top-left corner
    assert m["center"] == pytest.approx((CENTER[0] - x0, CENTER[1] - y0), abs=1.0)


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


WHITE = ((0, 0, 255), (0, 3, 254), (90, 5, 200))            # a blown-out spot: no color, hue is noise


@pytest.mark.software
def test_a_spot_with_no_color_gets_no_band():
    res = cl.combine([cl.measure_lamp(cl.traffic_roi(lamp_frame(*WHITE), SCENE_CONFIG))])
    assert "white, not colored" in cl.no_color(res)
    assert cl.no_color(cl.combine([measured("green", GREEN)[0]])) is None
    noisy = {**res, "s_median": 60.0, "hue_spread": 80.0}
    assert "isn't one lamp's color" in cl.no_color(noisy)


@pytest.mark.software
def test_two_lamps_found_at_the_same_spot_are_reported():
    at = lambda c: {"center": c}                                     # noqa: E731
    assert cl.same_spot({"red": at((91.8, 74.4)), "yellow": at((91.9, 74.2)), "green": at((94, 40))}) \
        == [("red", "yellow")]
    assert cl.same_spot({"red": at((90, 20)), "yellow": at((90, 40))}) == []


@pytest.mark.software
def test_the_command_line_writes_only_the_lamps_that_gave_a_band(tmp_path):
    out = tmp_path / "hsv.json"
    out.write_text(HSV_RANGES_PATH.read_text())
    before = json.loads(out.read_text())
    lines = []
    args = [f"red={write_frames(tmp_path, 'r', WHITE)}", f"green={write_frames(tmp_path, 'g', GREEN)}",
            f"--out={out}", "--write"]
    assert cl.main(args, config=SCENE_CONFIG, say=lines.append) == 0
    text = "\n".join(lines)
    assert "no band: the spot found is white" in text and "WARNING: red: no band suggested" in text
    after = json.loads(out.read_text())
    assert after["red_low"] == before["red_low"] and after["red_high"] == before["red_high"]
    assert after["green"] != before["green"]
    lines.clear()
    assert cl.main([args[0], f"--out={out}", "--write"], config=SCENE_CONFIG, say=lines.append) == 1
    assert "nothing written" in "\n".join(lines) and json.loads(out.read_text()) == after


def write_mixed(tmp_path, name, lamps):
    """A run folder whose frames show the given lamps, in order."""
    folder = tmp_path / name / "frames"
    folder.mkdir(parents=True)
    for i, lamp in enumerate(lamps):
        cv2.imwrite(str(folder / f"{i:06d}.png"), lamp_frame(*lamp))
    return tmp_path / name


@pytest.mark.software
def test_the_band_comes_from_the_frames_that_found_the_colored_lamp(tmp_path):
    out = tmp_path / "hsv.json"
    out.write_text(HSV_RANGES_PATH.read_text())
    lines = []
    folder = write_mixed(tmp_path, "r", [RED, WHITE, RED, RED])
    assert cl.main([f"red={folder}", f"--out={out}", "--write"], config=SCENE_CONFIG, say=lines.append) == 0
    text = "\n".join(lines)
    assert text.count("colored") >= 3 and "white / mixed" in text, "a line per frame"
    assert "only 3 of 4 frames found the colored lamp" in text
    written = json.loads(out.read_text())
    assert written["red_high"]["lower"][0] <= 176 and written["red_low"]["upper"][0] >= 2
    assert "blob area under it: " in text and "blob area under it: 0 " not in text


@pytest.mark.software
def test_with_half_the_frames_white_the_band_and_area_are_the_colored_frames_alone(tmp_path):
    """At exactly half, a median over every frame would land between the lamp and the white spot."""
    def run(lamps, name):
        out = tmp_path / f"{name}.json"
        out.write_text(HSV_RANGES_PATH.read_text())
        lines = []
        assert cl.main([f"red={write_mixed(tmp_path, name, lamps)}", f"--out={out}", "--write"],
                       config=SCENE_CONFIG, say=lines.append) == 0
        area = next(ln for ln in lines if "blob area under it" in ln)
        return json.loads(out.read_text()), area
    mixed, mixed_area = run([RED, WHITE, RED, WHITE], "mixed")
    clean, clean_area = run([RED, RED], "clean")
    assert (mixed["red_low"], mixed["red_high"]) == (clean["red_low"], clean["red_high"])
    assert mixed_area == clean_area


@pytest.mark.software
def test_the_lamps_hue_comes_from_its_colored_pixels_not_its_dark_edge():
    """A lamp disc with a dark, washed-out edge: that edge's hue is noise and mustn't widen the band or mark the frame mixed."""
    hsv = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    hsv[:] = (0, 0, 20)                                            # a dark background
    rng = np.random.default_rng(0)
    yy, xx = np.ogrid[:FRAME_H, :FRAME_W]
    edge = np.hypot(yy - CENTER[1], xx - CENTER[0]) <= LAMP_R + 4
    hsv[edge] = np.stack([rng.integers(0, 180, edge.sum()), rng.integers(0, 15, edge.sum()),
                          np.full(edge.sum(), 60)], 1).astype(np.uint8)    # unsaturated, of random hue
    cv2.circle(hsv, CENTER, 5, (86, 160, 230), -1)                 # the lamp: green, saturated
    m = cl.measure_lamp(cl.traffic_roi(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), SCENE_CONFIG))
    assert m["hue_spread"] <= 4, m["hue_spread"]
    e = cl.bands_for("green", m["band"])["green"]
    assert 80 <= e["lower"][0] <= 86 <= e["upper"][0] <= 92


@pytest.mark.software
def test_a_saturated_spot_of_many_hues_is_not_a_colored_lamp():
    assert cl.is_colored({"s_median": 200.0, "hue_spread": 10.0})
    assert not cl.is_colored({"s_median": 200.0, "hue_spread": 80.0})
    assert not cl.is_colored({"s_median": 10.0, "hue_spread": 5.0})


@pytest.mark.software
def test_under_half_the_frames_colored_is_no_band(tmp_path):
    lines = []
    folder = write_mixed(tmp_path, "r", [RED, WHITE, WHITE, WHITE])
    assert cl.main([f"red={folder}", f"--out={tmp_path / 'x.json'}"], config=SCENE_CONFIG, say=lines.append) == 0
    assert "no band:" in "\n".join(lines) and "red_low" not in "\n".join(lines)
    measures = [{"s_median": s, "hue_spread": 5.0} for s in (200, 200, 3, 3, 3)]
    assert cl.colored_frames(measures) is None
    assert len(cl.colored_frames(measures[:4])) == 2


@pytest.mark.software
def test_a_lamp_at_the_edge_of_the_traffic_roi_is_reported(tmp_path):
    m = {"center": (25.8, 0.4), "roi_size": (96, 108)}
    assert cl.at_edge(m) and not cl.at_edge({**m, "center": (48.0, 50.0)})
    lines = []
    folder = write_mixed(tmp_path, "r", [(RED[0], RED[1], RED[2])] * 2)
    top = tmp_path / "top" / "frames"
    top.mkdir(parents=True)
    for i in range(2):
        cv2.imwrite(str(top / f"{i:06d}.png"), lamp_frame(*RED, center=(216, round(TRAFFIC.y0 * FRAME_H) + 3)))
    assert cl.main([f"red={top.parent}", f"--out={tmp_path / 'x.json'}"], config=SCENE_CONFIG, say=lines.append) == 0
    assert "sits at the edge of the traffic ROI" in "\n".join(lines)
    lines.clear()
    cl.main([f"red={folder}", f"--out={tmp_path / 'x.json'}"], config=SCENE_CONFIG, say=lines.append)
    assert "edge of the traffic ROI" not in "\n".join(lines)


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


@pytest.mark.software
def test_each_frame_reports_its_clipped_core_and_a_lamp_that_doesnt_clip_is_warned(tmp_path):
    flat = (YELLOW[1], YELLOW[1], YELLOW[2])                        # the ring's color (S 140) right through: no clipped core
    lines = []
    args = [f"green={write_mixed(tmp_path, 'g', [GREEN, GREEN])}", f"yellow={write_mixed(tmp_path, 'y', [flat, flat])}",
            f"--out={tmp_path / 'x.json'}"]
    assert cl.main(args, config=SCENE_CONFIG, say=lines.append) == 0
    text = "\n".join(lines)
    assert "core" in text and " px  colored" in text
    assert "yellow: the lamp doesn't clip" in text and "green: the lamp doesn't clip" not in text
    m = cl.measure_lamp(cl.traffic_roi(lamp_frame(*GREEN), SCENE_CONFIG))
    assert m["core_px"] >= 25                                       # the drawn core: r 3, V 255, S 30


# =============================================================================
# Lit against off (an off= run)
# =============================================================================

# HSV: a lens's colored plastic unlit, and its LED lit: a white center, a colored
# ring and a dimmer colored glow past it, as the course's LEDs (2026-10-06)
LENS_OFF = {"red": (178, 200, 120), "green": (80, 200, 110)}
LENS_LIT = {"red": (178, 200, 245), "green": (80, 200, 245)}
GLOW_V = 170
BRIGHT_UNLIT_V = 215        # an unlit lens brighter than the lit LED's glow: V can't separate them
LENS_AT = {"red": (196, 50), "green": (236, 50)}
BOARD_HSV = (85, 200, 100)          # teal, under the lenses


def light(lit=None, unlit_like=None, brighter=0, red_box=False):
    """
    A teal board with a red and a green lens; lit= lights one; unlit_like=
    draws that unlit lens bright; red_box= adds a bright red box beside the
    light, there lit or not.
    """
    hsv = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    hsv[:] = (0, 0, 60 + brighter)
    cv2.rectangle(hsv, (180, 38), (252, 62), BOARD_HSV, -1)
    if red_box:
        cv2.rectangle(hsv, (160, 70), (172, 82), (178, 220, 240), -1)
    for color, at in LENS_AT.items():
        if color == lit:
            cv2.circle(hsv, at, 7, (LENS_LIT[color][0], 200, GLOW_V), -1)
            cv2.circle(hsv, at, 5, LENS_LIT[color], -1)
            cv2.circle(hsv, at, 2, (0, 0, 255), -1)                 # the white center
        elif color == unlit_like:
            cv2.circle(hsv, at, 5, (LENS_OFF[color][0], 200, BRIGHT_UNLIT_V), -1)
        else:
            cv2.circle(hsv, at, 5, LENS_OFF[color], -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def write_light(tmp_path, name, n=3, **kw):
    folder = tmp_path / name / "frames"
    folder.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"{i:06d}.png"), light(**kw))
    return folder.parent


def on_off(tmp_path, *labels):
    (tmp_path / "x.json").write_text(HSV_RANGES_PATH.read_text())     # --write merges into a full file
    lines = []
    code = cl.main([*labels, f"--out={tmp_path / 'x.json'}"], config=SCENE_CONFIG, say=lines.append)
    return code, "\n".join(lines)


@pytest.mark.software
def test_the_lamps_pixels_are_what_lights_up_between_the_off_run_and_its_run():
    on = [cl.traffic_roi(light("red"), SCENE_CONFIG)] * 2
    off = [cl.traffic_roi(light(), SCENE_CONFIG)] * 2
    found = cl.regions(cl.lit_pixels(on, off))
    x0 = TRAFFIC.x0 * FRAME_W
    assert len(found) == 1 and found[0][0] == pytest.approx(LENS_AT["red"][0] - x0, abs=1.5)
    assert found[0][1] == pytest.approx(LENS_AT["red"][1], abs=1.5)


@pytest.mark.software
def test_v_floors_sit_between_each_lamp_lit_and_off(tmp_path):
    red, green, off = (write_light(tmp_path, "r", lit="red"), write_light(tmp_path, "g", lit="green"),
                       write_light(tmp_path, "o"))
    code, out = on_off(tmp_path, f"red={red}", f"green={green}", f"off={off}", "--write")
    assert code == 0 and "no band" not in out
    bands = load_hsv_ranges(str(tmp_path / "x.json"))
    for color, entry in (("red", bands.red_high), ("green", bands.green)):
        assert LENS_OFF[color][2] < entry.lower[2] < LENS_LIT[color][2], color
    assert "0 px^2 off (the largest)" in out and "with the lamp off its band still finds" not in out
    assert "(+3 from the other lamps' runs)" in out


@pytest.mark.software
def test_an_unlit_lens_as_bright_as_a_lit_one_gets_no_band_and_points_at_the_white_center(tmp_path):
    red, off = write_light(tmp_path, "r", lit="red"), write_light(tmp_path, "o", unlit_like="red")
    code, out = on_off(tmp_path, f"red={red}", f"off={off}", "--write")
    assert "no S or V floor separates the lamp lit from unlit" in out and "its white center does" in out
    assert code == 1 and "nothing written" in out          # the only lamp gave no band


@pytest.mark.software
def test_the_whole_scene_brightening_is_reported(tmp_path):
    red, off = write_light(tmp_path, "r", lit="red", brighter=60), write_light(tmp_path, "o")
    _, out = on_off(tmp_path, f"red={red}", f"off={off}")
    assert "the scene changed between the runs" in out


@pytest.mark.software
def test_off_alone_is_refused_and_off_is_a_label():
    assert cl._parse(["red=a", "off=b"]) == {"red": "a", "off": "b"}
    lines = []
    assert cl.main(["off=x"], config=SCENE_CONFIG, say=lines.append) == 2


@pytest.mark.software
def test_cross_check_with_nothing_lit_is_none():
    roi = cl.traffic_roi(light(), SCENE_CONFIG)
    assert cl.cross_check([roi], [roi], np.zeros(roi.shape[:2], bool)) is None


@pytest.mark.software
def test_something_the_band_finds_with_the_lamp_off_is_reported(tmp_path):
    red, off = write_light(tmp_path, "r", lit="red", red_box=True), write_light(tmp_path, "o", red_box=True)
    _, out = on_off(tmp_path, f"red={red}", f"off={off}")
    assert "lit up: " in out and "with the lamp off its band still finds a blob" in out


def patch(bgr_hsv, size=(20, 20)):
    """A ROI of one HSV color."""
    return cv2.cvtColor(np.full((*size, 3), bgr_hsv, np.uint8), cv2.COLOR_HSV2BGR)


@pytest.mark.software
def test_floors_are_halfway_and_come_from_the_lamps_colored_pixels_and_the_unlit_ones_of_its_hue():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    lit = patch((178, 200, 240))
    lit[8:12, 8:12] = (255, 255, 255)                                   # the white center: no hue, no S
    off = patch((178, 150, 140))
    cc = cl.cross_check([lit], [off], mask)
    assert cc["s"]["on"] >= cl.MIN_LAMP_S                               # the white center isn't counted
    assert cc["v"]["floor"] == pytest.approx((cc["v"]["on"] + cc["v"]["off"]) / 2)
    assert cc["v"]["off"] < cc["v"]["floor"] < cc["v"]["on"] and cc["band"]["v_min"] == cc["v"]["floor"]
    assert cc["white"] == {"on": 16.0, "off": 0.0}
    other_hue = patch((90, 220, 250))                                   # bright, but no red could pass for it
    cc = cl.cross_check([lit], [other_hue], mask)
    assert cc["v"]["off"] is None and cc["v"]["floor"] is None and cc["band"]["v_min"] == cc["v"]["on"]
