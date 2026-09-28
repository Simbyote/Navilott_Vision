"""
test_debug_stopline.py  --  src/debugger/debug_stopline.py

The stop-line view is tested on real chains over the shared synthetic
scenes, headless: what it reads from the chain, how it grades each top edge,
that every state renders, the CSV row and the summary. Also the lane view's
labelling of candidates lane_offset skipped as part of a stop line.

--software  extract / grade / render / row / report, the live_view command
            line with --views stopline, and debug_lane's "stop_line" gate. No camera.
"""
import re
from dataclasses import replace

import cv2
import pytest

import src.debugger.live_view as lv
from src.debugger.debug_lane import candidate_gates
from src.debugger.debug_stopline import StopLineView
from src.phase2_linker import run_chain, run_live_view
from src.tests.scenes import SCENE_CONFIG, SCENES, SYNTHETIC_GROUND


# The lane detector's horizontal-line filter keeps stop lines out of the lane
# candidates, so lane_offset's stop-line check is a backstop. These tests of
# the "lane skip" marks turn the filter off to reach it
NO_LANE_FILTER = replace(SCENE_CONFIG, geometry=replace(
    SCENE_CONFIG.geometry, lane=replace(SCENE_CONFIG.geometry.lane, horizontal_edge_deg=None)))

def chain(scene, trace=True, config=NO_LANE_FILTER):
    return run_chain(SCENES[scene], 7, 350, config, trace=trace)

def data(scene, view=None, config=NO_LANE_FILTER):
    return (view or StopLineView()).extract(chain(scene, config=config))


# =============================================================================
# What the view reads
# =============================================================================

@pytest.mark.software
def test_extract_reads_the_candidates_the_measurement_and_the_trace():
    c = chain("stop_line_wide")
    d = StopLineView().extract(c)
    assert (d["frame_id"], d["timestamp_ms"]) == (7, 350)
    assert d["accepted"] == c.geometry.stop_line_candidates and len(d["accepted"]) == 1
    assert d["measured"] is c.stop_line and d["measured"].detected
    assert len(d["trace"]) == d["counts"]["seen"] == 1
    assert d["roi"].shape == c.roi.lane_roi.shape and d["top"].shape == d["roi"].shape


@pytest.mark.software
def test_extract_finds_the_lane_candidates_lane_offset_skipped():
    c = chain("stop_line_wide")
    skipped = StopLineView().extract(c)["lane_skipped"]
    horizontal = [cand.bbox for cand in c.geometry.lane_candidates if cand.bbox[2] >= cand.bbox[3]]
    assert skipped == horizontal and len(skipped) == 1
    assert StopLineView().extract(chain("two_boundary"))["lane_skipped"] == []


@pytest.mark.software
def test_each_top_edge_is_graded_pass_low_or_rejected_with_its_gate():
    view = StopLineView(conf_threshold=0.62)
    sm = view._summary(data("stop_line_tilted", view))         # conf 0.617: accepted but below
    assert (sm["passed"], sm["low"], sm["rejected"]) == (0, 1, 0)
    sm = view._summary(data("stop_line_wide", view))           # 0.667
    assert (sm["passed"], sm["low"], sm["rejected"]) == (1, 0, 0)
    blob = view._summary(data("horizontal_blob", view))
    assert blob["rejected"] == 1 and blob["entries"][0]["gate"] == "short"


# =============================================================================
# Drawing
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("scene", ["stop_line_wide", "stop_line_clipped", "stop_line_tilted",
                                   "horizontal_blob", "two_stop_lines", "stop_line_thick_marks",
                                   "two_boundary"])
@pytest.mark.parametrize("scale", [1, 2])
def test_every_state_renders_a_color_image_as_wide_as_the_zoomed_roi(scene, scale):
    view = StopLineView(conf_threshold=0.62)
    d = data(scene, view)
    img = view.render(d, scale)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[1] == d["roi"].shape[1] * scale * view.zoom


@pytest.mark.software
@pytest.mark.parametrize("scene", ["stop_line_wide", "stop_line_tilted", "stop_line_clipped"])
def test_the_band_spans_the_line_between_its_top_and_bottom_edges(scene):
    c = data(scene)["accepted"][0]
    s, y0 = 3, 40
    (tl, tr, br, bl) = StopLineView._band(c, s, y0).reshape(-1, 2)
    assert (tl[0], tr[0]) == (round(c.x_left * s), round(c.x_right * s))
    mid_top, mid_bottom = (tl[1] + tr[1]) / 2, (bl[1] + br[1]) / 2
    assert mid_top == pytest.approx(y0 + c.y_top_px * s, abs=2)
    assert mid_bottom == pytest.approx(y0 + c.y_bottom_px * s, abs=2)       # the ROI bottom when clipped
    assert mid_bottom - mid_top == pytest.approx((c.y_bottom_px - c.y_top_px) * s, abs=3)


@pytest.mark.software
def test_without_a_trace_accepted_lines_still_draw_and_the_footer_says_so():
    view = StopLineView()
    d = data("stop_line_wide", view)
    d["trace"] = None
    assert view._summary(d)["passed"] == 1
    assert view.render(d).shape[2] == 3


@pytest.mark.software
def test_a_chain_without_a_lane_roi_renders_a_notice():
    view = StopLineView()
    d = data("stop_line_wide", view)
    d["roi"] = None
    assert view.render(d).shape[2] == 3


# =============================================================================
# CSV row and summary
# =============================================================================

@pytest.mark.software
def test_row_matches_the_header_and_carries_the_measurement():
    view = StopLineView()
    row = dict(zip(StopLineView.CSV_FIELDS, view.row(data("stop_line_wide", view))))
    assert len(row) == len(StopLineView.CSV_FIELDS)
    assert (row["measured"], row["distance_px"], row["clipped"], row["lane_skipped"]) == (1, 26.0, 0, 1)
    assert row["thickness_px"] == pytest.approx(5.4, abs=0.5)
    assert (row["distance_cm"], row["proximity"]) == ("", pytest.approx(55.0 / 81.0, abs=1e-3))
    empty = dict(zip(StopLineView.CSV_FIELDS, view.row(data("two_boundary", view))))
    assert (empty["measured"], empty["distance_px"]) == (0, "")
    # With a ground plane the row carries the floor distance, and the header shows it
    grounded = replace(SCENE_CONFIG, ground=SYNTHETIC_GROUND)
    d = view.extract(run_chain(SCENES["stop_line_wide"], 7, 350, grounded, trace=True))
    row = dict(zip(StopLineView.CSV_FIELDS, view.row(d)))
    assert row["distance_cm"] == d["measured"].distance_cm and row["distance_cm"] > 0
    assert view.render(d).shape[2] == 3


@pytest.mark.software
def test_report_counts_measured_frames_distances_skips_and_rejections():
    view = StopLineView()
    for scene in ("stop_line_wide", "stop_line_clipped", "horizontal_blob", "two_boundary"):
        view.observe(data(scene, view))
    text = "\n".join(view.report())
    assert "[STOP LINE] 4 frames" in text
    assert re.search(r"frames with a measured line\s+2\b", text)
    assert "min 0.0" in text and "max 26.0" in text
    assert "skipped as part of a stop line  1" in text
    assert "short" in text


# =============================================================================
# live_view and the lane view
# =============================================================================

@pytest.mark.software
def test_live_view_runs_the_stopline_view_and_writes_its_video_csv_and_summary(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    for i, scene in enumerate(("two_boundary", "stop_line_wide", "stop_line_clipped")):
        cv2.imwrite(str(frames / f"{i:06d}.png"), SCENES[scene])
    out = tmp_path / "out"
    assert "stopline" in lv.VIEWS
    assert lv.cli(run_live_view, ["--frames", str(frames), "--no-display", "--views", "stopline",
                                  "--stopline-threshold", "0.99", "--out", str(out)]) == 0
    assert (out / "run_stopline.avi").stat().st_size > 0
    header, *rows = (out / "run_stopline.csv").read_text().splitlines()
    assert header.split(",") == list(StopLineView.CSV_FIELDS) and len(rows) == 3
    # 0.99 is above every line: the command line's threshold reaches the view, so both go amber
    graded = [dict(zip(StopLineView.CSV_FIELDS, r.split(","))) for r in rows]
    assert [(g["passed"], g["low"]) for g in graded] == [("0", "0"), ("0", "1"), ("0", "1")]
    assert "[STOP LINE] 3 frames" in (out / "summary.txt").read_text()


@pytest.mark.software
def test_the_lane_view_labels_a_candidate_lane_offset_skipped_as_on_the_stop_line():
    c = chain("stop_line_wide")
    gates = dict(candidate_gates(c.geometry, SCENE_CONFIG.lane_offset))
    skipped = [cand.bbox for cand in c.geometry.lane_candidates if cand.bbox[2] >= cand.bbox[3]]
    assert [gates[b] for b in skipped] == ["stop_line"]
    assert sum(g is None for g in gates.values()) == 2          # both lane lines stay usable
