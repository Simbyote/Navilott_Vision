"""
test_debug_maneuver.py  --  src/debugger/debug_maneuver.py

The after-the-run renderer: the maneuver strip for every step, and
render_run() on a real run folder written by maneuver_linker over the
simulated robot, including frames the recorder never wrote and a
records.pkl cut short by a crash.

--software  Drawing and rendering from files. No hardware.
"""
import os

import cv2
import numpy as np
import pytest

import src.debugger.debug_video as dv
from src.debugger.debug_maneuver import (
    STEP_COLORS, as_result, draw_strip, frame_path, read_records, render_run,
)
from src.debugger.debug_phase3 import Phase3View
from src.maneuver import ABORTED, FORWARD_1, SETTLE, TURN, ManeuverConfig
from src.tests.scenes import SCENE_CONFIG, SCENES
from src.tests.test_maneuver_linker import CFG, trial

W = 480


def record(step, **kw):
    base = {"t": 1.5, "step": step, "cmd_left": 0.4, "cmd_right": 0.41, "c_counts": 0.01,
            "c_heading": -0.005, "left_cps": 400.0, "right_cps": 395.0, "yaw_corrected": 0.2,
            "leg_progress": "", "turn_deg": "", "heading_deg": "", "event": ""}
    return {**base, **kw}


def near(img, color, tol=40):
    return int((np.abs(img.astype(int) - color).sum(axis=2) < tol).sum())


# =============================================================================
# The strip
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("rec", [record(SETTLE), record(FORWARD_1, leg_progress=150.0, heading_deg=-0.5),
                                 record(TURN, turn_deg=90.0), record(ABORTED, event="ABORT stall"),
                                 record("", t="", cmd_left="", cmd_right="")])
@pytest.mark.parametrize("scale", [1, 2])
def test_every_step_draws_one_strip_size(rec, scale):
    img = draw_strip(rec, ManeuverConfig(), W * scale, scale)
    assert img.shape == (4 * 16 * scale, W * scale, 3)


@pytest.mark.software
def test_the_leg_bar_fills_to_the_leg_progress():
    cfg = ManeuverConfig(leg_counts=400)
    half = draw_strip(record(FORWARD_1, leg_progress=200.0), cfg, W)
    full = draw_strip(record(FORWARD_1, leg_progress=400.0), cfg, W)
    assert near(full, dv.C_USABLE) > near(half, dv.C_USABLE) * 1.7


@pytest.mark.software
def test_the_turn_bar_shows_the_pass_band_and_the_angle():
    cfg = ManeuverConfig()
    img = draw_strip(record(TURN, turn_deg=120.0), cfg, W)
    assert near(img, (0, 90, 0), 10) > 0                     # the pass band
    assert near(img, STEP_COLORS[TURN]) > near(draw_strip(record(TURN, turn_deg=30.0), cfg, W), STEP_COLORS[TURN])


@pytest.mark.software
def test_an_abort_event_is_red():
    assert near(draw_strip(record(ABORTED, event="ABORT frame gap"), ManeuverConfig(), W), dv.C_RED) > 0


# =============================================================================
# render_run on a real run folder
# =============================================================================

@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    rep, out, *_ = trial(tmp_path_factory.mktemp("m"), render=False)
    return rep, out


def frames_in(path):
    cap, n, shapes = cv2.VideoCapture(str(path)), 0, set()
    while True:
        ok, f = cap.read()
        if not ok:
            return n, shapes
        n, _ = n + 1, shapes.add(f.shape)


@pytest.mark.software
def test_render_run_writes_one_frame_per_record_at_one_size(run_dir, tmp_path):
    rep, out = run_dir
    res = render_run(str(out), SCENE_CONFIG.lane_offset, CFG, 25.0)
    n, shapes = frames_in(out / "maneuver.avi")
    assert res["rendered"] == n == rep["run"]["frames"] and res["missing"] == 0 and len(shapes) == 1
    # Each image is the Phase 3 view with the strip under it (checked before the codec, which
    # rounds odd heights down)
    seen = []
    render_run(str(out), SCENE_CONFIG.lane_offset, CFG, 25.0, on_frame=seen.append)
    view = Phase3View(SCENE_CONFIG.lane_offset)
    first = next(read_records(str(out)))
    top = view.render(view.extract(as_result(first, SCENES["two_boundary"]), SCENES["two_boundary"]))
    assert seen[0].shape[0] == top.shape[0] + 4 * 16 and seen[0].shape[1] == top.shape[1]
    assert any(line.startswith("[PHASE 3]") for line in res["report"])
    header, *rows = (out / "maneuver_video.csv").read_text().splitlines()
    assert len(rows) == n


@pytest.mark.software
def test_missing_frames_are_skipped_and_counted(run_dir):
    rep, out = run_dir
    gone = [frame_path(str(out), r["frame_id"]) for r in list(read_records(str(out)))[5:8]]
    saved = [open(p, "rb").read() for p in gone]
    for p in gone:
        os.remove(p)
    try:
        res = render_run(str(out), SCENE_CONFIG.lane_offset, CFG, 25.0)
        assert res["missing"] == 3 and res["rendered"] == rep["run"]["frames"] - 3
    finally:
        for p, data in zip(gone, saved):
            open(p, "wb").write(data)


@pytest.mark.software
def test_a_records_file_cut_short_by_a_crash_renders_what_it_has(run_dir, tmp_path):
    rep, out = run_dir
    data = (out / "records.pkl").read_bytes()
    cut = tmp_path / "cut"
    cut.mkdir()
    (cut / "records.pkl").write_bytes(data[: len(data) // 2])
    os.symlink(out / "frames", cut / "frames")
    res = render_run(str(cut), SCENE_CONFIG.lane_offset, CFG, 25.0)
    assert 0 < res["rendered"] < rep["run"]["frames"]


@pytest.mark.software
def test_an_empty_run_folder_renders_nothing(tmp_path):
    assert render_run(str(tmp_path), SCENE_CONFIG.lane_offset, CFG, 25.0)["rendered"] == 0
