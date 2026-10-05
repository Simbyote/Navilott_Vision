"""
test_debug_phase3.py  --  src/debugger/debug_phase3.py

The Phase 3 view is tested on real Phase3Results (run_chain on the shared
synthetic scenes, then TracedPhase3Processor), headless: the frame size,
which must not change from frame to frame or the video writer drops frames,
the edge cases (no detections, no lane result, None distances, the first
frame), detection boxes by fate, the timeline window, the CSV row and the
summary.

--software  extract / observe / render / row / report. No camera.
"""
import copy
from dataclasses import replace

import pytest

import src.debugger.debug_video as dv
from src.debugger.debug_lane import HEADER_H
from src.debugger.debug_phase3 import Phase3View, TIMELINE_S
from src.debugger.estimation_debug import NO_RESULT, TracedPhase3Processor
from src.phase3_linker import Phase3Result, run_phase3_chain
from src.tests.scenes import SCENE_CONFIG, SCENES

LANE = SCENE_CONFIG.lane_offset

def results(*names, processor=None):
    """Phase3Results for a run of scenes, one frame each, 50 ms apart."""
    processor = processor or TracedPhase3Processor()
    return [run_phase3_chain(SCENES[n], i, i * 50, processor, None, SCENE_CONFIG)
            for i, n in enumerate(names)]

def view(fps=20):
    return Phase3View(LANE, fps)

def shown(v, res, scale=1):
    """extract, observe and render one result, as the linker does."""
    d = v.extract(res)
    v.observe(d)
    return d, v.render(d, scale)

# A traffic light and a stop sign as TracedPhase3Processor records them, placed in the frame
LIGHT = {"label": "red", "confidence": 0.62, "gate": 0.40, "passed": True,
         "bbox": (10, 10, 12, 12), "source_rect": (200, 0, 80, 60)}
SIGN = {"label": "stop", "confidence": 0.31, "gate": 0.45, "passed": False,
        "bbox": (5, 5, 20, 20), "source_rect": (360, 60, 100, 100)}

def with_detections(res):
    dbg = copy.deepcopy(res.p3_debug)
    dbg["traffic"]["detections"] = [LIGHT]
    dbg["stop_sign"]["detections"] = [SIGN]
    return replace(res, p3_debug=dbg)


# =============================================================================
# Size
# =============================================================================

@pytest.mark.software
def test_the_frame_is_as_wide_as_the_camera_frame_and_one_height_for_every_state():
    v = view()
    rs = results("two_boundary", "blind", "stop_line_wide", "two_boundary", "sign_and_lights")
    shapes = {shown(v, r)[1].shape for r in rs}
    assert len(shapes) == 1
    (h, w, c), = shapes
    assert (w, c) == (SCENES["two_boundary"].shape[1], 3)
    assert h > SCENES["two_boundary"].shape[0] + HEADER_H


@pytest.mark.software
def test_scale_2_is_twice_scale_1():
    r, = results("two_boundary")
    a = view().render(view().extract(r), 1)
    b = view().render(view().extract(r), 2)
    assert b.shape[:2] == (2 * a.shape[0], 2 * a.shape[1])


# =============================================================================
# Edge cases
# =============================================================================

@pytest.mark.software
def test_the_first_frame_renders_with_an_empty_timeline():
    r, = results("two_boundary")
    v = view()
    d = v.extract(r)
    assert len(v.history) == 0
    assert v.render(d).shape[2] == 3


@pytest.mark.software
def test_no_detections_draws_no_boxes():
    r, = results("two_boundary")
    assert r.p3_debug["traffic"]["detections"] == r.p3_debug["stop_sign"]["detections"] == []
    shown(view(), r)


@pytest.mark.software
def test_no_lane_result_reads_as_no_result():
    processor = TracedPhase3Processor()
    chain = results("two_boundary", processor=TracedPhase3Processor())[0].chain
    phase2 = replace(chain.phase2, lane_offset_results=[])
    packet, dbg = processor.process(phase2)
    res = Phase3Result(chain, packet, dbg, {"capture": 0.0, "phase2": 1.0, "phase3": 0.1, "total": 1.1})
    v = view()
    d, img = shown(v, res)
    assert d["dbg"]["lane"]["reason"] == NO_RESULT and d["dbg"]["lane"]["raw_offset"] is None
    assert v.history[-1]["raw"] is None
    row = dict(zip(Phase3View.CSV_FIELDS, v.row(d)))
    assert (row["lane_mode"], row["lane_raw"], row["lane_reason"]) == ("", "", NO_RESULT)


@pytest.mark.software
def test_none_distances_render_and_leave_the_csv_blank():
    v = view()
    for r in results("two_boundary", "stop_line_wide"):
        d, _ = shown(v, r)
    sl = d["dbg"]["stop_line"]
    assert sl["seen"] and not sl["state"] and sl["reported_px"] is None     # seen once: the vote hasn't flipped
    row = dict(zip(Phase3View.CSV_FIELDS, v.row(d)))
    assert (row["line_reported_px"], row["line_reported_cm"]) == ("", "")
    assert row["line_measured_px"] == sl["measured_px"] > 0


@pytest.mark.software
def test_a_held_distance_renders_with_px_and_cm():
    v = view()
    rs = results("stop_line_wide", "stop_line_wide", "two_boundary")
    for r in rs:
        d, _ = shown(v, r)
    sl = d["dbg"]["stop_line"]
    assert sl["held"] and sl["reported_px"] == rs[1].chain.stop_line.distance_px
    dbg = copy.deepcopy(d["dbg"])
    dbg["stop_line"]["reported_cm"] = 41.5                  # as with a ground homography
    assert v.render({**d, "dbg": dbg}).shape == v.render(d).shape


# =============================================================================
# What it draws
# =============================================================================

def _near(img, color, box, tol=30):
    """Pixels close to color inside box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    patch = img[y0:y1, x0:x1].astype(int)
    return int((abs(patch - color).sum(axis=2) < tol).sum())

@pytest.mark.software
@pytest.mark.parametrize("scale", [1, 2])
def test_a_passed_detection_is_boxed_green_and_a_gated_one_amber(scale):
    r, = results("two_boundary")
    v = view()
    img = v.render(v.extract(with_detections(r)), scale)
    s = scale
    y = HEADER_H                                                      # the frame sits under the header
    light = (210 * s - 3, (y + 10) * s - 3, 222 * s + 4, (y + 22) * s + 4)   # source_rect + bbox, with the outline
    sign = (365 * s - 3, (y + 65) * s - 3, 385 * s + 4, (y + 85) * s + 4)
    assert _near(img, dv.C_USABLE, light) > 10 and _near(img, dv.C_AMBER, light) == 0
    assert _near(img, dv.C_AMBER, sign) > 10 and _near(img, dv.C_USABLE, sign) == 0
    bare = v.render(v.extract(r), scale)
    assert _near(bare, dv.C_USABLE, light) == 0 and _near(bare, dv.C_AMBER, sign) == 0


@pytest.mark.software
def test_the_timeline_keeps_the_last_five_seconds():
    v = view(fps=4)
    assert v.history.maxlen == int(4 * TIMELINE_S)
    r, = results("two_boundary")
    for _ in range(50):
        v.observe(v.extract(r))
    assert len(v.history) == 20


@pytest.mark.software
def test_the_lane_bar_is_colored_by_status():
    rs = results("two_boundary", "blind")
    v = view()
    top_h = rs[0].chain.frame.frame.shape[0] + HEADER_H
    for r, status in zip(rs, ("vision", "hold")):
        d, img = shown(v, r)
        assert d["packet"].lane_status == status
        bg = tuple(int(c) for c in img[top_h + 2, 2])
        assert bg == {"vision": (30, 70, 30), "hold": (0, 70, 95)}[status]
    # "none" reports offset 0.0 with nothing behind it: not plotted as a raw measurement
    assert d["dbg"]["lane"]["raw_offset"] == 0.0 and v.history[-1]["raw"] is None


# =============================================================================
# CSV row and summary
# =============================================================================

@pytest.mark.software
def test_row_matches_the_header_and_carries_the_decisions():
    v = view()
    for r in results("two_boundary", "blind"):
        d, _ = shown(v, r)
    row = dict(zip(Phase3View.CSV_FIELDS, v.row(d)))
    assert len(row) == len(Phase3View.CSV_FIELDS) == len(v.row(d))
    assert (row["lane_status"], row["lane_accepted"], row["lane_reason"]) == ("hold", 0, "unusable_mode")
    assert (row["missed"], row["hold_max"]) == (1, 7)
    assert (row["traffic_buffer"], row["drive_state"], row["sign_buffer"]) == ("GG", "go", "FF")
    d = v.extract(with_detections(results("two_boundary")[0]))
    row = dict(zip(Phase3View.CSV_FIELDS, v.row(d)))
    assert row["traffic_detections"] == "red:0.620:pass" and row["sign_detections"] == "stop:0.310:gated"


@pytest.mark.software
def test_report_counts_status_reasons_and_holds():
    v = view()
    for r in results("two_boundary", "blind", "blind", "two_boundary"):
        shown(v, r)
    v.observe(v.extract(with_detections(results("two_boundary")[0])))
    text = "\n".join(v.report())
    assert "[PHASE 3] 5 frames" in text
    assert "unusable_mode" in text
    assert "traffic 0  stop sign 1" in text
