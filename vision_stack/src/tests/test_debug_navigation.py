"""
test_debug_navigation.py  --  src/debugger/debug_navigation.py

The navigation strip (one size for every kind of frame, the duty bars on
the right side of center and in the right color, DRIVE green and BRAKE red)
and render_run() on a real run folder from navigation_linker.
"""
import os

import cv2
import numpy as np
import pytest

import src.debugger.debug_video as dv
from src.debugger.debug_maneuver import frame_path, read_records
from src.debugger.debug_navigation import STRIP_LINES, draw_strip, render_run
from src.tests.scenes import SCENE_CONFIG
from src.tests.test_navigation_linker import go

W = 480
LH = 16
DRIVE = {"t": 1.25, "cmd_left": 0.35, "cmd_right": 0.45, "brake": 0, "reason": "steer", "source": "offset",
         "steer": 0.05, "lane_status": "vision", "lane_offset": 0.1, "lane_offset_cm": None,
         "heading_error": 0.0, "drive_state": "go", "stop_sign": 0, "stop_line_cm": None,
         "left_cps": 1100.0, "right_cps": 1200.0, "latency_ms": 42.0, "event": ""}
BRAKED = {**DRIVE, "cmd_left": 0.0, "cmd_right": 0.0, "brake": 1, "reason": "stop_sign", "source": "none",
          "steer": 0.0, "stop_sign": 1, "event": "-> stop_sign"}
REVERSE = {**DRIVE, "cmd_left": -0.3, "cmd_right": 0.7, "lane_offset_cm": 4.2, "stop_line_cm": 12.5}


def has(img, color, rows=None, cols=None, tol=40):
    """Any pixel within tol of color per channel (text is anti-aliased, so exact matches are rare)."""
    region = img if rows is None else img[rows]
    region = region if cols is None else region[:, cols]
    diff = np.abs(region.astype(int) - np.array(color, int))
    return bool(np.all(diff <= tol, axis=-1).any())


@pytest.mark.software
@pytest.mark.parametrize("n", [DRIVE, BRAKED, REVERSE, {}], ids=["drive", "brake", "reverse", "empty"])
@pytest.mark.parametrize("scale", [1, 2])
def test_every_frame_draws_one_strip_size(n, scale):
    assert draw_strip(n, W * scale, scale).shape == (STRIP_LINES * LH * scale, W * scale, 3)


@pytest.mark.software
def test_drive_is_green_and_brake_is_red_on_the_first_line():
    first = slice(0, LH)
    assert has(draw_strip(DRIVE, W), dv.C_USABLE, first) and not has(draw_strip(DRIVE, W), dv.C_RED, first)
    assert has(draw_strip(BRAKED, W), dv.C_RED, first)


@pytest.mark.software
def test_forward_duty_fills_right_of_center_in_green():
    img = draw_strip(DRIVE, W)
    bars = slice(2 * LH + 3, 3 * LH - 3)
    left_half = img[bars, : W // 2]
    mid = (6 + 16 + W // 2 - 6) // 2                     # the left bar's center line
    assert has(left_half, dv.C_USABLE, cols=slice(mid + 2, W // 2))
    assert not has(left_half, dv.C_USABLE, cols=slice(0, mid - 1))


@pytest.mark.software
def test_reverse_duty_fills_left_of_center_in_red():
    img = draw_strip(REVERSE, W)
    bars = slice(2 * LH + 3, 3 * LH - 3)
    left_half = img[bars, : W // 2]
    mid = (6 + 16 + W // 2 - 6) // 2
    assert has(left_half, dv.C_RED, cols=slice(0, mid - 1)) and not has(left_half, dv.C_USABLE)


@pytest.mark.software
def test_a_braked_frame_draws_no_duty_bars():
    bars = slice(2 * LH + 3, 3 * LH - 3)
    img = draw_strip(BRAKED, W)
    assert not has(img, dv.C_USABLE, bars) and not has(img, dv.C_RED, bars)


# =============================================================================
# render_run on a real run folder
# =============================================================================

@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    rep, out, *_ = go(tmp_path_factory.mktemp("n"), cam={"end_at": 11})
    return rep, out


@pytest.mark.software
def test_render_run_writes_one_frame_per_record_with_the_strip_under_it(run_dir):
    rep, out = run_dir
    seen = []
    res = render_run(str(out), SCENE_CONFIG.lane_offset, 20.0, on_frame=seen.append)
    assert res["rendered"] == len(seen) == rep["run"]["frames"] and res["missing"] == 0
    assert len({img.shape for img in seen}) == 1
    cap, n = cv2.VideoCapture(str(out / "nav.avi")), 0
    while cap.read()[0]:
        n += 1
    assert n == res["rendered"]
    assert any(line.startswith("[PHASE 3]") for line in res["report"])
    assert len((out / "nav_video.csv").read_text().splitlines()) == n + 1


@pytest.mark.software
def test_the_strip_is_the_last_rows_of_each_image(run_dir):
    rep, out = run_dir
    seen = []
    render_run(str(out), SCENE_CONFIG.lane_offset, 20.0, on_frame=seen.append)
    first = next(read_records(str(out)))
    strip = draw_strip(first["nav"], seen[0].shape[1])
    assert np.array_equal(seen[0][-strip.shape[0]:], strip)


@pytest.mark.software
def test_missing_frames_are_skipped_and_counted(run_dir):
    rep, out = run_dir
    gone = [frame_path(str(out), r["frame_id"]) for r in list(read_records(str(out)))[2:4]]
    saved = [open(p, "rb").read() for p in gone]
    for p in gone:
        os.remove(p)
    try:
        res = render_run(str(out), SCENE_CONFIG.lane_offset, 20.0)
        assert res["missing"] == 2 and res["rendered"] == rep["run"]["frames"] - 2
    finally:
        for p, data in zip(gone, saved):
            open(p, "wb").write(data)


@pytest.mark.software
def test_an_empty_run_folder_renders_nothing(tmp_path):
    assert render_run(str(tmp_path), SCENE_CONFIG.lane_offset, 20.0)["rendered"] == 0


@pytest.mark.software
@pytest.mark.parametrize("n", [DRIVE, BRAKED], ids=["drive", "brake"])
def test_the_first_line_names_the_deciding_rule_and_phase(n):
    first = slice(0, LH)
    plain = draw_strip(n, W)[first]
    assert not np.array_equal(draw_strip({**n, "rule": "stop_sign"}, W)[first], plain)
    assert not np.array_equal(draw_strip({**n, "rule": "stop_sign", "phase": "crossing"}, W)[first],
                              draw_strip({**n, "rule": "stop_sign"}, W)[first])


@pytest.mark.software
def test_the_first_line_names_the_route_step_and_the_intersection_stage():
    first = slice(0, LH)
    base = {**DRIVE, "rule": "intersection", "phase": "crossing"}
    with_step = draw_strip({**base, "step": "1/3 left", "maneuver": "left"}, W)[first]
    assert not np.array_equal(with_step, draw_strip(base, W)[first])
    turning = draw_strip({**base, "step": "1/3 left", "maneuver": "left", "stage": "turn"}, W)[first]
    assert not np.array_equal(turning, with_step)
