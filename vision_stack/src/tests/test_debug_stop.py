"""
test_debug_stop.py  --  src/debugger/debug_stop.py

Picture and statistics tests use hand-built view data (the dict extract()
returns), so every contour sits at a known pixel with a known state; the
extract tests feed the view real run_chain() output from a synthetic frame with
lane tape and a red octagon in the sign ROI.

--software  extract, grading into pass / low / reject, labels, render, CSV row
            and summary report. No camera.
--hardware  Records the stop view over a run (live camera or --replay) as
            video + CSV, and saves the rendered view for three sample frames.
"""
import re
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.debugger.debug_video as dv
from src.capture.camera import FrameData
from src.debugger.debug_stop import StopView
from src.params import STOP_SIGN
from src.perception.geometry import SignCandidate
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import ROIConfig, crop_rois
from src.config import MEASURED
from src.phase2_linker import run_chain
from src.tests.scenes import SCENE_CONFIG, synthetic_frame

ROI_SHAPE = (80, 120)                    # (h, w) of the hand-built sign ROI


def sign_entry(gate=None, conf=0.6, bbox=(20, 15, 40, 40), vertices=8, solidity=0.95, area=1200.0, poly=None):
    """One trace entry, as geometry's sign trace records it."""
    return {"bbox": bbox, "gate": gate, "area": area, "vertices": vertices,
            "solidity": solidity, "confidence": conf, "poly": poly}


def view_data(entries=(), trace=True, accepted=(), fused=None, suppressed=0, counts=None, fid=3, ts=150):
    """Hand-built extract() output on a dark color sign ROI with no redness."""
    return {"fused": fused, "suppressed": suppressed, "frame_id": fid, "timestamp_ms": ts,
            "roi": np.full(ROI_SHAPE + (3,), 30, np.uint8), "redness": np.zeros(ROI_SHAPE, np.uint8),
            "mask": np.zeros(ROI_SHAPE, np.uint8), "threshold": 20.0,
            "trace": list(entries) if trace else None, "accepted": list(accepted),
            "counts": counts or {}}


def box_corner(view, data, bbox, scale=1):
    """The rendered pixel at a bbox's bottom-right corner, clear of labels and polygon."""
    img = view.render(data, scale)
    s, _, _, _, hh, _ = view._metrics(scale)
    x, y, w, h = bbox
    return tuple(img[hh + (y + h) * s - 1, (x + w) * s - 1])


def sign_scene(frame_id=11, ts=222, sign=True):
    """Chain result for a frame with lane tape and, optionally, an octagon in the sign ROI."""
    frame = synthetic_frame([150, 290])
    if sign:
        sx, sy, sw, sh = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig()).sign_rect
        ox, oy, r = sx + sw // 2, sy + sh // 2, int(0.3 * min(sw, sh))
        pts = np.array([(ox + r * np.cos(np.pi / 8 + 2 * np.pi * k / 8),
                         oy + r * np.sin(np.pi / 8 + 2 * np.pi * k / 8)) for k in range(8)], np.int32)
        cv2.fillPoly(frame, [pts], (40, 40, 200))                           # stop-sign red, BGR
    return run_chain(frame, frame_id, ts, SCENE_CONFIG, trace=True), frame


@pytest.mark.software
def test_extract_reads_the_sign_branch_and_fusion_from_a_real_chain():
    chain, frame = sign_scene()
    data = StopView().extract(chain, frame)
    assert (data["frame_id"], data["timestamp_ms"]) == (11, 222)
    assert data["roi"].ndim == 3 and data["redness"].shape == data["mask"].shape == data["roi"].shape[:2]
    assert data["threshold"] >= SCENE_CONFIG.geometry.sign.min_redness
    assert len(data["accepted"]) == 1 and len(data["fused"]) == 1
    assert data["fused"][0].type == STOP_SIGN
    assert data["trace"] is not None and data["counts"]["accepted"] == 1


@pytest.mark.software
def test_extract_tells_no_fusion_apart_from_fusion_that_kept_nothing():
    # None: the process callable has no fusion; []: fusion ran and kept no stop sign
    chain, _ = sign_scene(sign=False)
    assert StopView().extract(chain)["fused"] == []
    bare = SimpleNamespace(geometry=chain.geometry, sign_debug=chain.sign_debug)
    assert StopView().extract(bare)["fused"] is None


@pytest.mark.software
def test_extract_counts_suppressed_stop_signs_from_the_fusion_log():
    chain, _ = sign_scene()
    log = [f"[SUPPRESSED] {STOP_SIGN}: conf=0.2", f"[SUPPRESSED] {STOP_SIGN}: conf=0.1", "[DISCARD] x"]
    chain = replace(chain, fusion_debug={"log": log})
    assert StopView().extract(chain)["suppressed"] == 2


@pytest.mark.software
def test_without_a_trace_the_accepted_candidates_stand_in():
    cand = SignCandidate(label=STOP_SIGN, bbox=(5, 5, 30, 30), contour=None, vertex_count=8,
                         confidence=0.7, frame_id=1, timestamp_ms=2, area=900.0, solidity=0.9)
    [e] = StopView()._entries(view_data(trace=False, accepted=[cand]))
    assert e["gate"] is None and (e["bbox"], e["vertices"], e["confidence"]) == ((5, 5, 30, 30), 8, 0.7)


@pytest.mark.software
@pytest.mark.parametrize("scale, zoom", [(1, 2), (2, 1), (1, 3)])
def test_render_is_both_panels_side_by_side_at_scale_times_zoom(scale, zoom):
    v = StopView(zoom=zoom)
    s, _, _, _, hh, fh = v._metrics(scale)
    h, w = ROI_SHAPE
    assert v.render(view_data(), scale).shape == (hh + h * s + fh, 2 * w * s + dv.GAP_PX * s, 3)


@pytest.mark.software
def test_render_without_a_sign_roi_draws_a_placeholder():
    data = view_data()
    data["roi"] = None
    assert StopView().render(data).shape[2] == 3


@pytest.mark.software
@pytest.mark.parametrize("thr, e, color", [
    (None, sign_entry(conf=0.3), dv.C_USABLE),
    (0.5, sign_entry(conf=0.7), dv.C_USABLE),
    (0.5, sign_entry(conf=0.3), dv.C_AMBER),
    (None, sign_entry(gate="vertices", vertices=5), dv.C_RED),
])
def test_each_contour_is_boxed_in_its_state_color(thr, e, color):
    assert box_corner(StopView(thr), view_data([e]), e["bbox"]) == color


@pytest.mark.software
def test_render_leaves_the_chain_images_alone():
    data = view_data([sign_entry()])
    data["mask"][20:60, 30:90] = 255
    before = {k: data[k].copy() for k in ("roi", "redness", "mask")}
    StopView().render(data)
    assert all(np.array_equal(data[k], before[k]) for k in before)


@pytest.mark.software
@pytest.mark.parametrize("e, label", [
    (sign_entry(gate="vertices", vertices=5), "vert 5"),
    (sign_entry(gate="solidity", solidity=0.612), "sol 0.61"),
    (sign_entry(gate="area", area=37.4), "area 37"),
    (sign_entry(gate="not_largest", area=412.6), "smaller 413"),
    (sign_entry(gate="hull"), "hull"),
    (sign_entry(conf=0.684, vertices=8), "v8 c0.68"),
])
def test_labels_name_the_failing_gate_and_value(e, label):
    v = StopView()
    assert v._label(e, v._state(e)) == label


@pytest.mark.software
def test_low_label_shows_the_threshold_it_missed():
    v = StopView(0.7)
    e = sign_entry(conf=0.55)
    assert v._label(e, v._state(e)) == "v8 c0.55<0.70"


@pytest.mark.software
def test_row_matches_the_csv_header_and_reports_the_best_and_fused_sign():
    fused = [SimpleNamespace(confidence=0.81)]
    data = view_data([sign_entry(conf=0.4), sign_entry(conf=0.8), sign_entry("area", 0.9)],
                     fused=fused, counts={"seen": 3, "area": 1, "accepted": 2})
    row = dict(zip(StopView.CSV_FIELDS, StopView(0.5).row(data)))
    assert len(row) == len(StopView.CSV_FIELDS)
    assert (row["traced"], row["passed"], row["low"], row["best_conf"]) == (3, 1, 1, 0.8)
    assert (row["rej_area"], row["fused"], row["fused_conf"]) == (1, 1, 0.81)


@pytest.mark.software
def test_row_leaves_fusion_columns_blank_without_fusion():
    row = dict(zip(StopView.CSV_FIELDS, StopView().row(view_data())))
    assert row["fused"] == "" and row["fused_conf"] == "" and row["best_conf"] == ""


@pytest.mark.software
def test_report_counts_frames_and_totals_rejections_by_gate():
    v = StopView(0.5)
    v.observe(view_data([sign_entry(conf=0.7)], fused=[SimpleNamespace(confidence=0.7)],
                        counts={"seen": 2, "vertices": 1, "accepted": 1}))
    v.observe(view_data([sign_entry(conf=0.3)], fused=[], counts={"seen": 1, "accepted": 1}))
    v.observe(view_data([], fused=[], counts={"seen": 4, "area": 3, "hull": 1}))
    report = "\n".join(v.report())
    assert "[STOP SIGN] 3 frames" in report
    assert re.search(r"through the gates\s+2\s+\(\s*66\.7%\)", report)   # 2 of 3 frames had a candidate
    assert re.search(r"threshold 0\.50\s+1\s", report)                  # only one passed it
    assert re.search(r"passed a stop sign on\s+1\s", report)
    assert re.search(r"area\s+3\n", report) and re.search(r"vertices\s+1\n", report)
    assert "seen" not in report.split("by gate:")[1]           # bookkeeping keys aren't gates


@pytest.mark.software
def test_report_omits_the_threshold_and_fusion_lines_it_has_no_data_for():
    v = StopView()
    v.observe(view_data([sign_entry()]))
    report = "\n".join(v.report())
    assert "threshold" not in report.split("would pass")[0] and "fusion" not in report


@pytest.mark.hardware
def test_stop_view_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    view = StopView()
    writer = dv.ViewWriter(str(artifacts.path / "stop_view.avi"), StopView.CSV_FIELDS)
    try:
        for i, fd in enumerate(frames(n)):
            chain = run_chain(fd.frame, fd.frame_id, fd.timestamp_ms, MEASURED, trace=True)
            data = view.extract(chain, fd.frame)
            view.observe(data)
            img = writer.push(view.render(data), view.row(data))
            if i in (0, n // 2, n - 1):
                artifacts.image(f"{fd.frame_id:06d}_stop_view.png", img)
    finally:
        writer.close()
    if not writer.frames_written:
        pytest.skip("no frames delivered")
    (artifacts.path / "stop_summary.txt").write_text("\n".join(view.report()) + "\n")

@pytest.mark.software
def test_the_red_mask_is_outlined_on_the_redness_panel_and_the_threshold_named():
    from src.debugger.debug_stop import C_MASK
    data = view_data()
    view = StopView()
    s, _, _, _, hh, _ = view._metrics(1)
    right = lambda img: img[hh:hh + ROI_SHAPE[0] * s, ROI_SHAPE[1] * s + dv.GAP_PX * s:]
    assert not (right(view.render(data)) == C_MASK).all(axis=2).any()
    data["mask"][20:60, 30:90] = 255
    assert (right(view.render(data)) == C_MASK).all(axis=2).sum() > 100


@pytest.mark.software
def test_a_gray_roi_still_renders():
    data = view_data()
    data["roi"] = np.full(ROI_SHAPE, 30, np.uint8)
    assert StopView().render(data).shape[2] == 3


@pytest.mark.software
def test_row_carries_the_not_largest_count_and_the_redness_threshold():
    data = view_data(counts={"seen": 3, "not_largest": 2, "accepted": 1})
    data["threshold"] = 62.0
    row = dict(zip(StopView.CSV_FIELDS, StopView().row(data)))
    assert (row["rej_not_largest"], row["red_threshold"]) == (2, 62.0)
