"""
test_debug_lanegeo.py  --  src/debugger/debug_lanegeo.py and geometry's lane trace

The lane-geometry view is tested on real chains over the shared synthetic
scenes, headless: geometry's per-contour lane trace, what the view reads
from the chain, how it sorts contours into geometry-refused, lane_offset-
refused and usable, that every state renders, the CSV row and the summary.

--software  trace / extract / render / row / report, and the live_view
            command line with --views lanegeo. No camera.
"""
import re
from dataclasses import replace

import cv2
import pytest

import src.debugger.debug_video as dv
import src.debugger.live_view as lv
from src.debugger.debug_lanegeo import GEO_GATES, LaneGeometryView
from src.perception.geometry import extract_lane_candidates
from src.phase2_linker import run_chain, run_live_view
from src.tests.scenes import LANE_RECT, SCENE_CONFIG, SCENES, same, scene


LANE = SCENE_CONFIG.lane_offset
# lane_offset's stop-line check is a backstop behind the horizontal filter;
# turning the filter off reaches it (as in test_debug_stopline)
NO_LANE_FILTER = replace(SCENE_CONFIG, geometry=replace(
    SCENE_CONFIG.geometry, lane=replace(SCENE_CONFIG.geometry.lane, horizontal_edge_deg=None)))
# 28 px tape crossed by a stop line: below it only single tape edges survive,
# and the thinnest ones are refused by geometry (too_few_pts)
WIDE_TAPE_STOP = scene(marks=(150, 290), mark_width=24, stop_line=(100, 340, 40), stop_line_thickness=8)

def chain(frame, trace=True, config=SCENE_CONFIG):
    if isinstance(frame, str):
        frame = SCENES[frame]
    return run_chain(frame, 7, 350, config, trace=trace)

def data(frame, trace=True, config=SCENE_CONFIG, view=None):
    return (view or LaneGeometryView(lane_config=config.lane_offset)).extract(chain(frame, trace, config))


# =============================================================================
# geometry's lane trace
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("frame", ["two_boundary", "noise_b", "horizontal_blob", "stop_line_touching",
                                   "dashed", WIDE_TAPE_STOP])
def test_the_trace_has_one_entry_per_contour_agreeing_with_the_reject_counts(frame):
    dbg = chain(frame).lane_debug
    trace, rc = dbg["trace"], dbg["reject_counts"]
    assert len(trace) == rc["seen"]
    for gate in (*GEO_GATES, None):
        assert sum(e["gate"] == gate for e in trace) == rc["accepted" if gate is None else gate]
    for e in trace:
        assert e["bbox"] == cv2.boundingRect(e["contour"])


@pytest.mark.software
def test_the_trace_records_what_the_gate_measured():
    refused = [e for e in chain(WIDE_TAPE_STOP).lane_debug["trace"] if e["gate"] is not None]
    assert refused and all(e["gate"] == "too_few_pts" for e in refused)
    assert all(e["value"] == len(e["contour"]) < 5 for e in refused)


@pytest.mark.software
def test_each_gate_is_named_in_the_trace():
    # SCENE_CONFIG's lane gates are lenient; tighten aspect and intensity so a
    # square blob and a dim bar each trip one
    frame = scene()
    x0, y0 = LANE_RECT[:2]
    cv2.rectangle(frame, (x0 + 200, y0 + 30), (x0 + 224, y0 + 54), (240, 240, 240), -1)
    cv2.rectangle(frame, (x0 + 60, y0 + 10), (x0 + 65, y0 + 60), (150, 150, 150), -1)
    lane = replace(SCENE_CONFIG.geometry.lane, min_aspect=2.0, min_intensity=180.0)
    config = replace(SCENE_CONFIG, geometry=replace(SCENE_CONFIG.geometry, lane=lane))
    by_x = {e["bbox"][0] // 10: e for e in chain(frame, config=config).lane_debug["trace"]}
    assert (by_x[19]["gate"], by_x[19]["value"]) == ("aspect", pytest.approx(1.0, abs=0.1))
    assert by_x[5]["gate"] == "intensity" and by_x[5]["value"] < 180
    assert by_x[14]["gate"] is None and by_x[28]["gate"] is None


@pytest.mark.software
def test_without_trace_there_is_no_trace_and_the_candidates_are_unchanged():
    c_on, c_off = chain("noise_b", trace=True), chain("noise_b", trace=False)
    assert "trace" not in c_off.lane_debug
    same(c_on.geometry.lane_candidates, c_off.geometry.lane_candidates)


@pytest.mark.software
def test_extract_lane_candidates_traces_only_when_asked():
    roi = cv2.cvtColor(SCENES["noise_b"], cv2.COLOR_BGR2GRAY)
    x, y, w, h = LANE_RECT
    roi = roi[y:y + h, x:x + w]
    geo = SCENE_CONFIG.geometry
    args = (roi, geo.canny, geo.lane, 0, 0)
    _, dbg = extract_lane_candidates(*args, draw_overlays=False)
    assert "trace" not in dbg
    _, dbg = extract_lane_candidates(*args, draw_overlays=False, trace=True)
    assert len(dbg["trace"]) == dbg["reject_counts"]["seen"]


# =============================================================================
# What the view reads
# =============================================================================

@pytest.mark.software
def test_extract_reads_the_roi_edges_trace_candidates_and_result():
    c = chain("two_boundary")
    d = LaneGeometryView(lane_config=LANE).extract(c)
    assert (d["frame_id"], d["timestamp_ms"]) == (7, 350)
    assert d["roi"].shape == c.roi.lane_roi.shape
    for k in ("edges_raw", "edges_lane", "edges"):
        assert d[k].shape == d["roi"].shape
    assert [cand for cand, _ in d["candidates"]] == c.geometry.lane_candidates
    assert d["offset"] is c.offset and d["trace"] is c.lane_debug["trace"]


@pytest.mark.software
def test_extract_needs_the_lane_config():
    with pytest.raises(ValueError, match="lane_config"):
        LaneGeometryView().extract(chain("two_boundary"))


@pytest.mark.software
def test_candidates_are_graded_with_lane_offset_gates_including_the_stop_line():
    usable = [g for _, g in data("two_boundary")["candidates"]]
    assert usable == [None, None]
    # With the filter off, the stop line itself is a lane candidate lane_offset skips
    gates = [g for _, g in data("stop_line_wide", config=NO_LANE_FILTER)["candidates"]]
    assert gates.count("stop_line") == 1 and gates.count(None) == 2
    # A tighter confidence gate refuses everything, and says so
    strict = LaneGeometryView(lane_config=replace(LANE, conf_threshold=0.99))
    assert {g for _, g in strict.extract(chain("two_boundary"))["candidates"]} == {"confidence"}


@pytest.mark.software
def test_removed_edge_px_counts_what_the_horizontal_filter_took_out():
    assert data("two_boundary")["removed_edge_px"] == 0
    d = data("stop_line_touching")
    assert d["removed_edge_px"] == int(((d["edges_raw"] > 0) & (d["edges_lane"] == 0)).sum()) > 100
    assert data("stop_line_touching", config=NO_LANE_FILTER)["removed_edge_px"] == 0


# =============================================================================
# Drawing
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("frame", ["two_boundary", "one_boundary", "blind", "noise_b", "horizontal_blob",
                                   "stop_line_touching", "stop_line_wide", WIDE_TAPE_STOP])
@pytest.mark.parametrize("scale", [1, 2])
def test_every_state_renders_a_color_image_as_wide_as_the_zoomed_roi(frame, scale):
    view = LaneGeometryView(lane_config=LANE)
    d = data(frame, view=view)
    img = view.render(d, scale)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[1] == d["roi"].shape[1] * scale * view.zoom


@pytest.mark.software
def test_each_stage_draws_in_its_own_color():
    def roi_panel_colors(view, d):
        """The colors with more than 20 px in the ROI panel (anti-aliasing shifts a line's edge pixels)."""
        _, _, _, _, hh, _ = view._metrics(1)
        img = view.render(d)[hh:hh + d["roi"].shape[0] * view.zoom].astype(int)  # the edge panel reuses red
        return {c for c in (dv.C_USABLE, dv.C_RED, dv.C_AMBER)
                if (abs(img - c).sum(axis=2) < 30).sum() > 20}
    view = LaneGeometryView(lane_config=LANE)
    assert roi_panel_colors(view, data("two_boundary", view=view)) == {dv.C_USABLE}
    # Tape edges refused by geometry, stubs of the stop line refused by lane_offset
    assert roi_panel_colors(view, data(WIDE_TAPE_STOP, view=view)) == {dv.C_USABLE, dv.C_RED, dv.C_AMBER}
    strict = LaneGeometryView(lane_config=replace(LANE, conf_threshold=0.99))
    assert roi_panel_colors(strict, data("two_boundary", view=strict)) == {dv.C_AMBER}


@pytest.mark.software
def test_without_a_trace_candidates_still_draw_and_the_footer_says_so():
    view = LaneGeometryView(lane_config=LANE)
    d = data("noise_b", trace=False, view=view)
    assert d["trace"] is None
    assert view.render(d).shape[2] == 3


@pytest.mark.software
def test_a_chain_without_a_lane_roi_renders_a_notice():
    view = LaneGeometryView(lane_config=LANE)
    d = data("two_boundary", view=view)
    d["roi"] = None
    assert view.render(d).shape[2] == 3


# =============================================================================
# CSV row and summary
# =============================================================================

@pytest.mark.software
def test_row_matches_the_header_and_carries_the_result():
    view = LaneGeometryView(lane_config=LANE)
    d = data(WIDE_TAPE_STOP, view=view)
    row = dict(zip(LaneGeometryView.CSV_FIELDS, view.row(d)))
    assert len(row) == len(LaneGeometryView.CSV_FIELDS)
    rc = d["counts"]
    assert (row["contours"], row["geo_rejected"], row["rej_too_few_pts"]) == (rc["seen"], 2, 2)
    assert (row["candidates"], row["mode"]) == (len(d["candidates"]), d["offset"].mode)
    assert row["usable"] + row["offset_rejected"] == row["candidates"]
    assert row["left_x"] == d["offset"].left_x and row["removed_edge_px"] > 0
    strict = LaneGeometryView(lane_config=replace(LANE, conf_threshold=0.99))
    row = dict(zip(LaneGeometryView.CSV_FIELDS, strict.row(data("two_boundary", view=strict))))
    assert (row["usable"], row["offset_rejects"]) == (0, "confidence:2")


@pytest.mark.software
def test_report_counts_modes_filtered_frames_and_rejections_by_stage():
    view = LaneGeometryView(lane_config=LANE)
    for frame in ("two_boundary", "stop_line_touching", WIDE_TAPE_STOP, "blind"):
        view.observe(data(frame, view=view))
    text = "\n".join(view.report())
    assert "[LANE GEOMETRY] 4 frames" in text
    assert re.search(r"horizontal-line filter removed edges\s+2\b", text)
    assert re.search(r"refused by geometry, by gate:\n\s+too_few_pts\s+2\b", text)
    assert "refused by lane_offset" in text


# =============================================================================
# live_view
# =============================================================================

@pytest.mark.software
def test_live_view_runs_the_lanegeo_view_and_writes_its_video_csv_and_summary(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    for i, frame in enumerate((SCENES["two_boundary"], SCENES["stop_line_touching"], WIDE_TAPE_STOP)):
        cv2.imwrite(str(frames / f"{i:06d}.png"), frame)
    out = tmp_path / "out"
    assert "lanegeo" in lv.VIEWS
    assert lv.cli(run_live_view, ["--frames", str(frames), "--no-display", "--views", "lanegeo",
                                  "--out", str(out)]) == 0
    assert (out / "run_lanegeo.avi").stat().st_size > 0
    header, *rows = (out / "run_lanegeo.csv").read_text().splitlines()
    assert header.split(",") == list(LaneGeometryView.CSV_FIELDS) and len(rows) == 3
    graded = [dict(zip(LaneGeometryView.CSV_FIELDS, r.split(","))) for r in rows]
    # live_view handed the view its lane config: candidates got lane_offset's gates
    assert all(int(g["usable"]) >= 1 for g in graded)
    assert int(graded[1]["removed_edge_px"]) > 0 and graded[0]["removed_edge_px"] == "0"
    assert "[LANE GEOMETRY] 3 frames" in (out / "summary.txt").read_text()
