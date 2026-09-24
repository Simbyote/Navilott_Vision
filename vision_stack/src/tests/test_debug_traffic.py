"""
test_debug_traffic.py  --  src/debugger/debug_traffic.py

Picture and statistics tests use hand-built view data, so every blob and mask
sits at a known pixel; the extract tests feed the view real run_chain() output
from a synthetic frame with a red light in the traffic ROI, using the same
explicit HSV bands as the color branch tests rather than the scaffold or a
calibration.

--software  extract (on and off), per-color totals, grading, mask panel,
            labels, CSV row and summary report. No camera.
--hardware  Records the traffic view over a run (live camera or --replay) as
            video + CSV, and saves the rendered view for three sample frames.
            Uses calibration/hsv_ranges.json if present, else the scaffold.
"""
import re
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.debugger.debug_video as dv
from src.capture.camera import FrameData
from src.debugger.debug_traffic import COLORS, LIGHT_COLORS, TrafficView
from src.params import GREEN, HSV_RANGES_PATH, RED, TRAFFIC_LIGHT, YELLOW
from src.perception.color_branch import BlobFilter, ColorConfig, ColorRange, HSVRanges, load_color_config
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import ROIConfig, crop_rois
from src.phase2_linker import MEASURED, run_chain, synthetic_frame

TEST_HSV = HSVRanges(
    red_low=ColorRange((0, 120, 120), (10, 255, 255)),
    red_high=ColorRange((170, 120, 120), (180, 255, 255)),
    yellow=ColorRange((20, 120, 120), (35, 255, 255)),
    green=ColorRange((40, 120, 120), (80, 255, 255)))
TEST_CFG = replace(MEASURED, color=ColorConfig(TEST_HSV, BlobFilter(min_area=50.0, max_area=5000.0,
                                                                   min_aspect=0.3, max_aspect=3.0,
                                                                   ref_area=800.0)))
ROI_SHAPE = (60, 90)                     # (h, w) of the hand-built traffic ROI


def blob(label=RED, gate=None, conf=0.5, bbox=(10, 10, 20, 20), area=300.0, aspect=1.0, fill=0.75, hsv=(2.0, 240.0, 230.0)):
    """One trace entry, as the color branch's blob trace records it."""
    return {"label": label, "bbox": bbox, "gate": gate, "area": area, "aspect": aspect,
            "fill": fill, "confidence": conf, "hsv": hsv}


def view_data(entries=(), trace=True, enabled=True, calibrated=True, masks=None, mask_px=None,
              counts=None, fused=None, suppressed=0, accepted=(), fid=4, ts=200):
    """Hand-built extract() output on a dark BGR traffic ROI."""
    h, w = ROI_SHAPE
    return {"frame_id": fid, "timestamp_ms": ts, "enabled": enabled, "calibrated": calibrated,
            "roi": np.full((h, w, 3), 20, np.uint8),
            "masks": masks or {c: np.zeros(ROI_SHAPE, np.uint8) for c in COLORS},
            "mask_px": mask_px or {c: 0 for c in COLORS}, "counts": counts or {},
            "trace": list(entries) if trace else None, "accepted": list(accepted),
            "fused": fused, "suppressed": suppressed}


def light_scene(color=(0, 0, 255), frame_id=11, ts=222, cfg=TEST_CFG):
    """Chain result for a frame with lane tape and one filled circle in the traffic ROI."""
    frame = synthetic_frame([150, 290])
    tx, ty, tw, th = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig()).traffic_rect
    cv2.circle(frame, (tx + tw // 2, ty + th // 2), int(0.1 * min(tw, th)), color, -1)
    return run_chain(frame, frame_id, ts, cfg, trace=True), frame


@pytest.mark.software
def test_extract_reads_the_color_branch_and_fusion_from_a_real_chain():
    chain, frame = light_scene()
    data = TrafficView().extract(chain, frame)
    assert (data["frame_id"], data["timestamp_ms"]) == (11, 222)
    assert data["enabled"] and data["roi"].ndim == 3
    assert set(data["masks"]) == set(COLORS) and data["mask_px"][RED] > 0
    assert [c.label for c in data["accepted"]] == [RED]
    assert len(data["fused"]) == 1 and data["fused"][0].type == TRAFFIC_LIGHT


@pytest.mark.software
def test_with_the_color_branch_off_the_view_reports_off_and_draws_a_notice():
    chain, frame = light_scene(cfg=MEASURED)                   # MEASURED leaves the branch off
    v = TrafficView()
    data = v.extract(chain, frame)
    assert not data["enabled"]
    v.observe(data)
    assert v.render(data).shape[2] == 3
    assert "color branch OFF" in "\n".join(v.report())


@pytest.mark.software
def test_extract_tells_no_fusion_apart_from_fusion_that_kept_nothing():
    chain, _ = light_scene(color=(255, 255, 255))              # white: no light detected
    assert TrafficView().extract(chain)["fused"] == []
    bare = SimpleNamespace(geometry=chain.geometry, traffic=chain.traffic, traffic_debug=chain.traffic_debug)
    assert TrafficView().extract(bare)["fused"] is None


@pytest.mark.software
def test_extract_counts_suppressed_lights_from_the_fusion_log():
    chain, _ = light_scene()
    log = [f"[SUPPRESSED] {TRAFFIC_LIGHT}: green", "[SUPPRESSED] stop_sign: x"]
    assert TrafficView().extract(replace(chain, fusion_debug={"log": log}))["suppressed"] == 1


@pytest.mark.software
def test_totals_sum_each_count_over_the_three_colors():
    counts = {RED: {"seen": 3, "area": 1, "aspect": 1, "accepted": 1},
              YELLOW: {"seen": 1, "area": 1, "aspect": 0, "accepted": 0},
              GREEN: {"seen": 2, "area": 0, "aspect": 0, "accepted": 2}}
    assert TrafficView()._totals(view_data(counts=counts)) == {"seen": 6, "area": 2, "aspect": 1, "accepted": 3}


@pytest.mark.software
def test_render_is_the_roi_panel_plus_a_mask_column_a_third_as_wide():
    v = TrafficView()
    s, _, _, _, hh, fh = v._metrics(1)
    h, w = ROI_SHAPE
    pw = w * s
    assert v.render(view_data()).shape == (hh + h * s + fh, pw + dv.GAP_PX * s + pw // 3, 3)


@pytest.mark.software
@pytest.mark.parametrize("i, color", list(enumerate(COLORS)))
def test_each_mask_is_tinted_in_its_lights_color_in_its_own_row(i, color):
    v = TrafficView()
    masks = {c: np.zeros(ROI_SHAPE, np.uint8) for c in COLORS}
    masks[color][:] = 255
    img = v.render(view_data(masks=masks))
    s, _, _, _, hh, _ = v._metrics(1)
    pw, ph = ROI_SHAPE[1] * s, ROI_SHAPE[0] * s
    x0, mh = pw + dv.GAP_PX * s, ph // 3
    probe_x = x0 + (pw // 3) // 2
    assert tuple(img[hh + i * mh + mh - 3, probe_x]) == LIGHT_COLORS[color]   # near the tile's bottom, clear of its label
    other = (i + 1) % 3
    assert tuple(img[hh + other * mh + mh - 3, probe_x]) == (0, 0, 0)


@pytest.mark.software
@pytest.mark.parametrize("thr, e, color", [
    (None, blob(conf=0.2), dv.C_USABLE),
    (0.5, blob(conf=0.3), dv.C_AMBER),
    (None, blob(gate="aspect", aspect=4.5), dv.C_RED),
])
def test_each_blob_is_boxed_in_its_state_color(thr, e, color):
    v = TrafficView(thr)
    img = v.render(view_data([e]))
    s, _, _, _, hh, _ = v._metrics(1)
    x, y, w, h = e["bbox"]
    assert tuple(img[hh + (y + h) * s - 1, (x + w) * s - 1]) == color


@pytest.mark.software
def test_uncalibrated_ranges_are_flagged_in_red_and_in_the_report():
    v = TrafficView()
    calibrated = v.render(view_data([blob()]))
    flagged = v.render(view_data([blob()], calibrated=False))
    _, _, _, _, hh, _ = v._metrics(1)
    changed = np.any(flagged != calibrated, axis=2)
    assert changed[:hh].any() and not changed[hh:].any()           # only the header changes
    reddish = flagged[changed].astype(int)
    assert (reddish[:, 2] > reddish[:, 1] + 50).any()               # antialiased, so no exact C_RED pixel
    v.observe(view_data([blob()], calibrated=False))
    assert "WARNING" in "\n".join(v.report())


@pytest.mark.software
@pytest.mark.parametrize("e, state, label", [
    (blob(gate="aspect", aspect=4.5), "reject", "red asp 4.50"),
    (blob(gate="area", area=12.6), "reject", "red area 13"),
    (blob(conf=0.34, fill=0.69), "pass", "red c0.34 f0.69"),
    (blob(conf=0.34, fill=None), "pass", "red c0.34"),
])
def test_labels_name_the_color_and_the_failing_gate_or_the_fill(e, state, label):
    assert TrafficView()._label(e, state) == label


@pytest.mark.software
def test_low_label_shows_the_threshold_it_missed():
    v = TrafficView(0.5)
    e = blob(conf=0.3, fill=None)
    assert v._label(e, v._state(e)) == "red c0.30<0.50"


@pytest.mark.software
def test_row_matches_the_csv_header_and_carries_best_blob_masks_and_fusion():
    fused = [SimpleNamespace(label_detail=GREEN, confidence=0.6)]
    data = view_data([blob(conf=0.3), blob(GREEN, conf=0.6, hsv=(60.0, 250.0, 200.0))],
                     mask_px={RED: 10, YELLOW: 0, GREEN: 40}, fused=fused,
                     counts={RED: {"seen": 1, "accepted": 1}, GREEN: {"seen": 2, "aspect": 1, "accepted": 1}})
    row = dict(zip(TrafficView.CSV_FIELDS, TrafficView().row(data)))
    assert len(row) == len(TrafficView.CSV_FIELDS)
    assert (row["best_label"], row["best_conf"], row["best_h"]) == (GREEN, 0.6, 60.0)
    assert (row["mask_red_px"], row["mask_green_px"], row["seen"], row["rej_aspect"]) == (10, 40, 3, 1)
    assert (row["fused"], row["fused_label"], row["fused_conf"]) == (1, GREEN, 0.6)


@pytest.mark.software
def test_row_without_hsv_or_fusion_leaves_those_columns_blank():
    row = dict(zip(TrafficView.CSV_FIELDS, TrafficView().row(view_data(trace=False))))
    assert (row["best_h"], row["fused"], row["fused_label"]) == ("", "", "")


@pytest.mark.software
def test_report_counts_candidates_fused_colors_coverage_and_rejects():
    v = TrafficView()
    h, w = ROI_SHAPE
    red_px = h * w // 10                                             # 10% of the ROI
    v.observe(view_data([blob(conf=0.6)], mask_px={RED: red_px, YELLOW: 0, GREEN: 0},
                        fused=[SimpleNamespace(label_detail=RED, confidence=0.6)],
                        counts={RED: {"seen": 2, "area": 1, "accepted": 1}}))
    v.observe(view_data([], mask_px={RED: 0, YELLOW: 0, GREEN: 0}, fused=[],
                        counts={RED: {"seen": 1, "aspect": 1}}))
    v.observe(view_data(enabled=False))                              # counted, but adds nothing else
    report = "\n".join(v.report())
    assert "[TRAFFIC LIGHT] 3 frames" in report
    assert re.search(r"through the blob filter\s+1\s", report)
    assert "fused light color: red 1" in report
    assert "median mask coverage of the ROI: red 10.0%" in report  # median of 10% and 0%, upper middle
    assert "blobs seen 3, accepted 1; rejected by gate: area 1, aspect 1" in report


def _color_config():
    """Calibrated ranges if the file exists, else the scaffold, so the view always draws."""
    if HSV_RANGES_PATH.exists():
        return load_color_config(str(HSV_RANGES_PATH))
    return ColorConfig(HSVRanges(), BlobFilter())


@pytest.mark.hardware
def test_traffic_view_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    cfg = replace(MEASURED, color=_color_config())
    view = TrafficView()
    writer = dv.ViewWriter(str(artifacts.path / "traffic_view.avi"), TrafficView.CSV_FIELDS)
    try:
        for i, fd in enumerate(frames(n)):
            chain = run_chain(fd.frame, fd.frame_id, fd.timestamp_ms, cfg, trace=True)
            data = view.extract(chain, fd.frame)
            view.observe(data)
            img = writer.push(view.render(data), view.row(data))
            if i in (0, n // 2, n - 1):
                artifacts.image(f"{fd.frame_id:06d}_traffic_view.png", img)
    finally:
        writer.close()
    if not writer.frames_written:
        pytest.skip("no frames delivered")
    (artifacts.path / "traffic_summary.txt").write_text("\n".join(view.report()) + "\n")