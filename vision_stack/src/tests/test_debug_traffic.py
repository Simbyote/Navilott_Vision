"""
test_debug_traffic.py  --  src/debugger/debug_traffic.py

Picture and statistics tests use hand-built view data, so every blob and mask
sits at a known pixel; the extract tests feed the view real run_chain() output
from a synthetic frame with a red light in the traffic ROI, using the same
explicit HSV bands as the color branch tests rather than the scaffold or a
calibration.

--software  extract (on and off), per-color totals, grading, mask panel,
            labels, CSV row and summary report; in glow mode the white mask
            tile, glow gate labels with the ring share, and glow gate counts
            in the footer, CSV and report. No camera.
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
from src.debugger.debug_traffic import COLORS, GLOW_GATES, LIGHT_COLORS, WHITE_TINT, TrafficView
from src.params import GREEN, HSV_RANGES_PATH, RED, TRAFFIC_LIGHT, YELLOW
from src.perception.color_branch import BlobFilter, ColorConfig, ColorRange, GlowFilter, HSVRanges, load_color_config
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import ROIConfig, crop_rois
from src.phase2_linker import run_chain
from src.tests.scenes import SCENE_CONFIG, synthetic_frame

TEST_HSV = HSVRanges(
    red_low=ColorRange((0, 120, 120), (10, 255, 255)),
    red_high=ColorRange((170, 120, 120), (180, 255, 255)),
    yellow=ColorRange((20, 120, 120), (35, 255, 255)),
    green=ColorRange((40, 120, 120), (80, 255, 255)))
TEST_CFG = replace(SCENE_CONFIG, color=ColorConfig(TEST_HSV, BlobFilter(min_area=50.0, max_area=5000.0,
                                                                   min_aspect=0.3, max_aspect=3.0,
                                                                   ref_area=800.0, min_roundness=0.0,
                                                                   min_core_px=0)))
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


# Lamp radius in px: ~200 px^2, over the test blob gates' min_area, and fits
# the default traffic ROI, 62 px tall since it was cut to the course's light (2026-10-06)
LAMP_R = 8


def light_scene(color=(0, 0, 255), frame_id=11, ts=222, cfg=TEST_CFG):
    """Chain result for a frame with lane tape and one filled circle in the traffic ROI."""
    frame = synthetic_frame([150, 290])
    tx, ty, tw, th = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig()).traffic_rect
    cv2.circle(frame, (tx + tw // 2, ty + th // 2), LAMP_R, color, -1)
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
def test_glow_mode_frames_extract_render_and_log():
    frame = synthetic_frame([150, 290])
    tx, ty, tw, th = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig()).traffic_rect
    center = (tx + tw // 2, ty + th // 2)
    cv2.circle(frame, center, 6, (0, 0, 255), -1)
    cv2.circle(frame, center, 3, (255, 255, 255), -1)                  # a lit LED: white center, red ring
    cfg = replace(TEST_CFG, color=replace(TEST_CFG.color, glow=GlowFilter()))
    chain = run_chain(frame, 11, 222, cfg, trace=True)
    v = TrafficView()
    data = v.extract(chain, frame)
    assert [c.label for c in data["accepted"]] == [RED] and data["trace"][0]["gate"] is None
    assert data["glow"] and np.count_nonzero(data["white"]) > 0
    assert v.render(data).ndim == 3
    row = dict(zip(v.CSV_FIELDS, v.row(data)))
    assert len(row) == len(v.CSV_FIELDS) and row["best_label"] == RED
    assert row["glow"] == 1 and row["mask_white_px"] > 0 and row["best_ring_share"] > 0.5
    v.observe(data)
    report = "\n".join(v.report())
    assert "fusion passed a light on" in report and "named by its ring (glow)" in report


@pytest.mark.software
def test_with_the_color_branch_off_the_view_reports_off_and_draws_a_notice():
    chain, frame = light_scene(cfg=replace(SCENE_CONFIG, color=ColorConfig()))    # no HSV ranges: branch off
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


# =============================================================================
# Glow mode
# =============================================================================

def spot(label=RED, gate=None, px=5, conf=1.0, bbox=(10, 10, 6, 6), ring=20, votes=None, aspect=1.0):
    """One glow trace entry, as _glow_candidates records it."""
    votes = votes if votes is not None else {RED: 5, YELLOW: 0, GREEN: 0}
    return {"label": label, "bbox": bbox, "gate": gate, "area": float(px), "aspect": aspect,
            "fill": None, "confidence": conf, "hsv": None, "ring": ring, "votes": votes}


def glow_data(entries=(), white=None, **kw):
    data = view_data(entries, **kw)
    data["glow"] = True
    data["white"] = white if white is not None else np.zeros(ROI_SHAPE, np.uint8)
    return data


@pytest.mark.software
def test_glow_render_stacks_the_white_mask_over_the_three_color_masks():
    v = TrafficView()
    white = np.full(ROI_SHAPE, 255, np.uint8)
    masks = {c: np.zeros(ROI_SHAPE, np.uint8) for c in COLORS}
    masks[GREEN][:] = 255
    img = v.render(glow_data(white=white, masks=masks))
    s, _, _, _, hh, fh = v._metrics(1)
    pw, ph = ROI_SHAPE[1] * s, ROI_SHAPE[0] * s
    x0, mh = pw + dv.GAP_PX * s, ph // 4
    assert img.shape == (hh + ph + fh, x0 + pw // 3, 3)                     # as wide as blob mode
    tw = round(mh * ROI_SHAPE[1] / ROI_SHAPE[0])                             # tiles keep the ROI's aspect
    assert tuple(img[hh + mh - 3, x0 + tw - 1]) == WHITE_TINT
    assert tuple(img[hh + mh - 3, x0 + tw + 1]) == (0, 0, 0)
    probe_x = x0 + tw // 2
    assert tuple(img[hh + mh - 3, probe_x]) == WHITE_TINT                    # tile 0: white
    assert tuple(img[hh + 4 * mh - 3, probe_x]) == LIGHT_COLORS[GREEN]       # tile 3: green
    assert tuple(img[hh + 2 * mh - 3, probe_x]) == (0, 0, 0)                 # tile 1: red, empty


@pytest.mark.software
@pytest.mark.parametrize("e, label", [
    (spot(gate="white", px=1, label="none"), "white 1px"),
    (spot(gate="shape", aspect=10.5), "red asp 10.50"),
    (spot(gate="ring", label="none", ring=50, votes={RED: 3, YELLOW: 0, GREEN: 0}), "no ring 6%"),
    (spot(gate="ring", label="none", votes={}), "no ring"),
    (spot(YELLOW, gate="smaller", px=2), "yellow 2px smaller"),
    (spot(conf=0.6, px=3, ring=20, votes={RED: 4, YELLOW: 1, GREEN: 0}), "red c0.60 3px ring 20%"),
])
def test_glow_labels_name_the_glow_gate_and_the_ring_share(e, label):
    v = TrafficView()
    assert v._label(e, v._state(e)) == label


@pytest.mark.software
def test_glow_low_label_shows_the_threshold_it_missed():
    v = TrafficView(0.5)
    e = spot(conf=0.4, px=2)
    assert v._label(e, v._state(e)) == "red c0.40 2px ring 25%<0.50"


@pytest.mark.software
def test_glow_row_and_report_count_the_glow_gates_from_the_trace():
    entries = [spot(conf=0.6, px=3, ring=20, votes={RED: 4, YELLOW: 0, GREEN: 0}),
               spot(YELLOW, gate="smaller", px=2), spot("none", gate="ring", votes={}),
               spot("none", gate="ring", votes={}), spot(gate="shape"), spot("none", gate="white", px=1)]
    white = np.zeros(ROI_SHAPE, np.uint8)
    white[5, 5:9] = 255
    data = glow_data(entries, white=white, counts={RED: {"seen": 2, "shape": 1, "accepted": 1}})
    v = TrafficView()
    row = dict(zip(v.CSV_FIELDS, v.row(data)))
    assert (row["glow"], row["mask_white_px"], row["best_ring_share"]) == (1, 4, 0.2)
    assert [row[f"rej_{g}"] for g in GLOW_GATES] == [1, 1, 2, 1]
    v.observe(data)
    v.observe(data)
    report = "\n".join(v.report())
    assert "white spots seen 12, accepted 2; rejected by gate: white 2, shape 2, ring 4, smaller 2" in report
    assert "through the blob filter" not in report


@pytest.mark.software
def test_glow_without_a_trace_counts_from_reject_counts_and_blob_rows_leave_glow_blank():
    data = glow_data(trace=False, counts={RED: {"seen": 2, "white": 1, "accepted": 1}})
    assert TrafficView._glow_totals(data) == {"seen": 2, "accepted": 1, "white": 1, "shape": 0,
                                               "ring": 0, "smaller": 0}
    row = dict(zip(TrafficView.CSV_FIELDS, TrafficView().row(view_data([blob()]))))
    assert row["glow"] == 0 and [row[f"rej_{g}"] for g in GLOW_GATES] == ["", "", "", ""]
    assert row["mask_white_px"] == "" and row["best_ring_share"] == ""


@pytest.mark.software
def test_glow_render_footer_and_header_differ_from_blob_mode():
    v = TrafficView()
    entries = [spot(conf=0.6, px=3)]
    glow_img = v.render(glow_data(entries))
    blob_img = v.render(view_data(entries))
    _, _, _, _, hh, _ = v._metrics(1)
    assert glow_img.shape == blob_img.shape                      # one size, so a run's video doesn't rescale
    assert (glow_img[:hh, :200] != blob_img[:hh, :200]).any()    # the best line reads white px and ring
    assert v.render(glow_data()).ndim == 3                      # no spot at all


def _color_config():
    """Calibrated ranges if the file exists, else the scaffold, so the view always draws."""
    if HSV_RANGES_PATH.exists():
        return load_color_config(str(HSV_RANGES_PATH))
    return ColorConfig(HSVRanges(), BlobFilter())


@pytest.mark.hardware
def test_traffic_view_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    cfg = replace(SCENE_CONFIG, color=_color_config())
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