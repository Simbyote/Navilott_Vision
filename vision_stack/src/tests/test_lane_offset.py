"""
test_lane_offset.py  --  src/perception/lane_offset.py

compute_lane_offset consumes GeometryBranchResult directly (a sibling of
fusion, not downstream of it), so these tests build LaneCandidate and
GeometryBranchResult by hand rather than chaining from run_geometry_stage.
The chained tests at the bottom cover the handoff from real geometry output.

Sign convention (from the module docstring): offset is (center_x - lane_center)
/ center_x. A lane center LEFT of ROI center (the robot has drifted right)
gives a POSITIVE offset; a lane center to the right gives a negative one.
Tests are written against this, not against intuition about "left is negative".

--software  Contract, gate, pairing-rule and known-answer tests. No camera.
--hardware  Times compute_lane_offset per frame (live or --replay, chained
            through preprocess -> crop -> geometry) and writes CSV, the debug
            log, and an anchor overlay.
"""
import time
from dataclasses import asdict, replace

import cv2
import numpy as np
import pytest

from src.capture.camera import FrameData
from src.perception import lane_offset as lo
from src.perception.geometry import GeometryBranchResult, GeometryConfig, LaneCandidate, run_geometry_stage
from src.perception.lane_offset import (
    BoundaryAnchor, LaneOffsetConfig, LaneOffsetResult,
    compute_lane_offset, foot_x,
)
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import ROIConfig, crop_rois
from src.params import FRAME_H, FRAME_W
from src.tests.artifacts import summarize

# Explicit rather than the shipped defaults, so retuning never breaks these tests
TEST_CFG = LaneOffsetConfig(
    conf_threshold=0.30, min_proximity=0.25, min_length_px=25.0,
    min_width_px=1.0, max_width_px=25.0, min_intensity=90.0,
    min_lane_width_px=60.0, max_lane_width_px=400.0,
    expected_half_lane_px=120.0, foot_band_px=6)
UNCALIBRATED = replace(TEST_CFG, expected_half_lane_px=None)

ROI_W = 300
CENTER_X = ROI_W / 2.0          # where the robot sits in the lane ROI
MODES = ("two_boundary", "left_only", "right_only", "single_uncalibrated", "none")


def make_roi(lane_rect=(0, 0, ROI_W, 50), frame_id=1, ts=2):
    """Minimal ROICropResult: compute_lane_offset only reads lane_rect and the stamp."""
    from src.perception.roi_crop import ROICropResult
    return ROICropResult(
        lane_roi=np.zeros((1, 1), np.uint8), traffic_roi=np.zeros((1, 1, 3), np.uint8),
        sign_roi=np.zeros((1, 1), np.uint8), lane_rect=lane_rect,
        traffic_rect=(0, 0, 1, 1), sign_rect=(0, 0, 1, 1),
        frame_id=frame_id, timestamp_ms=ts, source_shape=(FRAME_H, FRAME_W))


def cand(fx, confidence=0.9, proximity=0.9, length_px=100.0, width_px=8.0,
        mean_intensity=200.0, contour=None, foot=None, frame_id=1, ts=2):
    """Hand-built LaneCandidate. foot defaults to fx, so foot_x() reads the stored value."""
    x = int(round(fx - 4))
    return LaneCandidate(
        label="lane_boundary", bbox=(x, 20, 8, 20), contour=contour,
        confidence=confidence, frame_id=frame_id, timestamp_ms=ts,
        proximity=proximity, width_px=width_px, length_px=length_px,
        mean_intensity=mean_intensity, foot_x=fx if foot is None else foot)


def geo(cands, frame_id=1, ts=2, signs=()):
    """GeometryBranchResult from candidate lists."""
    return GeometryBranchResult(lane_candidates=list(cands), sign_candidates=list(signs),
                                frame_id=frame_id, timestamp_ms=ts)


def run(cands, cfg=TEST_CFG, roi=None, frame_id=1, ts=2):
    """compute_lane_offset on hand-built candidates sharing one frame stamp."""
    roi = roi or make_roi(frame_id=frame_id, ts=ts)
    return compute_lane_offset(geo(cands, frame_id, ts), roi, cfg)


def assert_offset_contract(result, dbg, roi):
    """Everything LaneOffsetResult and the debug summary document."""
    assert isinstance(result, LaneOffsetResult)
    assert (result.frame_id, result.timestamp_ms) == (roi.frame_id, roi.timestamp_ms)
    assert result.mode in MODES
    assert -1.0 <= result.offset <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.boundary_count >= 0
    assert (result.mode == "two_boundary") == (result.left_x is not None and result.right_x is not None)
    if result.lane_width_px is not None:
        assert result.lane_width_px == pytest.approx(result.right_x - result.left_x, abs=1e-6)
    for key in ("frame_id", "timestamp_ms", "mode", "raw_count", "usable_count", "anchors", "log"):
        assert key in dbg
    assert dbg["usable_count"] == result.boundary_count
    assert isinstance(dbg["log"], list)


@pytest.mark.software
def test_foot_x_prefers_the_stored_value():
    assert foot_x(cand(150.0, foot=150.0)) == 150.0


@pytest.mark.software
def test_foot_x_falls_back_to_the_contour_when_unstored():
    contour = np.array([[10, 10], [13, 10], [13, 90], [10, 90]], np.int32).reshape(-1, 1, 2)
    c = cand(0.0, contour=contour, foot=-1.0)
    assert foot_x(c, band_px=6) == 11.5          # mean x of the two lowest-row points


@pytest.mark.software
@pytest.mark.parametrize("contour", [None, np.empty((0, 1, 2), np.int32)])
def test_foot_x_falls_back_to_bbox_center_with_no_contour(contour):
    c = cand(0.0, contour=contour, foot=-1.0)     # bbox is (x=6, y=20, w=8, h=20) for fx=10
    assert foot_x(c) == c.bbox[0] + c.bbox[2] / 2.0


@pytest.mark.software
def test_foot_x_ignores_a_stored_negative_sentinel_not_just_exactly_minus_one():
    contour = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], np.int32).reshape(-1, 1, 2)
    assert foot_x(cand(0.0, contour=contour, foot=-0.5)) == 2.0


@pytest.mark.software
def test_foot_x_stored_at_exactly_zero_is_a_real_value_not_the_sentinel():
    # 0.0 is a legitimate anchor x (the left ROI edge); only < 0.0 means unset.
    # fx=200 puts the bbox center far from 0, so a fallback would be caught.
    assert foot_x(cand(200.0, foot=0.0)) == 0.0


@pytest.mark.software
@pytest.mark.parametrize("field, value, usable", [
    ("confidence", 0.29, False), ("confidence", 0.30, True),
    ("proximity", 0.24, False), ("proximity", 0.25, True),
    ("length_px", 24.9, False), ("length_px", 25.0, True),
    ("width_px", 0.99, False), ("width_px", 1.0, True),
    ("width_px", 25.0, True), ("width_px", 25.01, False),
    ("mean_intensity", 89.9, False), ("mean_intensity", 90.0, True),
], ids=lambda v: str(v))
def test_gate_edges_are_inclusive_on_the_documented_side(field, value, usable):
    log = []
    c = cand(100, **{field: value})
    assert lo._usable(c, TEST_CFG, log) is usable
    assert bool(log) == (not usable)
    if not usable:
        assert log[0].startswith("[REJECT]") and field.split("_")[0] in log[0]


@pytest.mark.software
def test_a_candidate_can_pass_geometry_and_still_fail_this_stricter_gate():
    # These gates are stricter than LaneContourFilter by design
    weak = cand(100, confidence=0.15)             # a low but nonzero geometry confidence
    log = []
    assert lo._usable(weak, TEST_CFG, log) is False


@pytest.mark.software
def test_anchor_weight_blends_confidence_and_proximity_without_replacing_confidence():
    a = lo._anchor(cand(100, confidence=0.8, proximity=0.6), TEST_CFG)
    assert a.weight == round(0.8 * (0.5 + 0.5 * 0.6), 4)
    assert a.foot_x == 100.0 and a.candidate.confidence == 0.8


@pytest.mark.software
def test_full_proximity_gives_full_weight_zero_proximity_gives_half():
    assert lo._anchor(cand(100, confidence=1.0, proximity=1.0), TEST_CFG).weight == 1.0
    assert lo._anchor(cand(100, confidence=1.0, proximity=0.0), TEST_CFG).weight == 0.5


def anc(fx, **kw):
    """BoundaryAnchor at foot x = fx, built through _anchor with TEST_CFG."""
    return lo._anchor(cand(fx, **kw), TEST_CFG)


@pytest.mark.software
def test_pair_straddling_center_takes_the_nearest_on_each_side():
    left, right = lo._lane_pair([anc(50), anc(200), anc(10), anc(280)], CENTER_X)
    assert (left.foot_x, right.foot_x) == (50.0, 200.0)


@pytest.mark.software
def test_pair_all_on_the_right_takes_the_two_nearest_center_in_x_order():
    left, right = lo._lane_pair([anc(160), anc(220), anc(300)], CENTER_X)
    assert (left.foot_x, right.foot_x) == (160.0, 220.0)


@pytest.mark.software
def test_pair_all_on_the_left_takes_the_two_nearest_center_in_x_order():
    left, right = lo._lane_pair([anc(50), anc(120), anc(10)], CENTER_X)
    assert (left.foot_x, right.foot_x) == (50.0, 120.0)


@pytest.mark.software
def test_pair_a_single_anchor_on_either_side_is_the_one_sided_case():
    assert lo._lane_pair([anc(200)], CENTER_X) == (None, lo._lane_pair([anc(200)], CENTER_X)[1])
    left, right = lo._lane_pair([anc(200)], CENTER_X)
    assert left is None and right.foot_x == 200.0
    left, right = lo._lane_pair([anc(50)], CENTER_X)
    assert right is None and left.foot_x == 50.0


@pytest.mark.software
def test_pair_of_no_anchors_is_none_none():
    assert lo._lane_pair([], CENTER_X) == (None, None)


@pytest.mark.software
def test_pair_exactly_at_center_counts_as_the_right_side():
    left, right = lo._lane_pair([anc(CENTER_X)], CENTER_X)
    assert left is None and right.foot_x == CENTER_X


@pytest.mark.software
def test_extreme_pair_would_give_a_different_answer_than_nearest_pair():
    # Taking the outermost anchors instead of the nearest-to-center pair
    # centers the robot on the street, not its own lane
    left, right = lo._lane_pair([anc(10), anc(100), anc(200), anc(290)], CENTER_X)
    assert (left.foot_x, right.foot_x) == (100.0, 200.0)
    assert (left.foot_x, right.foot_x) != (10.0, 290.0)


@pytest.mark.software
def test_symmetric_pair_around_center_gives_zero_offset():
    res, dbg = run([cand(100), cand(200)])
    assert res.mode == "two_boundary"
    assert res.offset == 0.0 and res.left_x == 100.0 and res.right_x == 200.0
    assert res.lane_width_px == 100.0 and res.boundary_count == 2
    assert_offset_contract(res, dbg, make_roi())


@pytest.mark.software
def test_lane_center_left_of_roi_center_gives_a_positive_offset():
    # center_x=150, lane_center=130 -> offset=(150-130)/150, per the sign convention
    res, _ = run([cand(80), cand(180)])
    assert res.offset == pytest.approx((CENTER_X - 130) / CENTER_X, abs=1e-4)
    assert res.offset > 0


@pytest.mark.software
def test_lane_center_right_of_roi_center_gives_a_negative_offset():
    res, _ = run([cand(120), cand(220)])
    assert res.offset < 0


@pytest.mark.software
def test_two_boundary_confidence_is_the_mean_of_the_two_weights():
    a, b = cand(100, confidence=0.4, proximity=0.6), cand(200, confidence=0.5, proximity=0.5)
    res, _ = run([a, b])
    wa, wb = lo._anchor(a, TEST_CFG).weight, lo._anchor(b, TEST_CFG).weight
    assert res.confidence == round((wa + wb) / 2, 4)


@pytest.mark.software
def test_offset_is_clamped_to_unit_range():
    res, _ = run([cand(0), cand(1)])                # lane_width=1 < min -> falls back
    assert -1.0 <= res.offset <= 1.0
    huge_roi = make_roi(lane_rect=(0, 0, 20, 50))    # center_x=10, anchors far outside it
    res, _ = compute_lane_offset(geo([cand(-500), cand(600)]), huge_roi, replace(TEST_CFG, max_lane_width_px=2000))
    assert -1.0 <= res.offset <= 1.0


@pytest.mark.software
def test_center_x_comes_from_lane_rect_width_not_the_rects_x_origin():
    roi_shifted = make_roi(lane_rect=(999, 0, ROI_W, 50))     # same width, different x origin
    res, _ = compute_lane_offset(geo([cand(100), cand(200)]), roi_shifted, TEST_CFG)
    assert res.offset == 0.0                                  # identical to the x=0 case


@pytest.mark.software
def test_anchors_too_close_together_merge_into_a_single_sided_read():
    a, b = cand(140, confidence=0.9), cand(160, confidence=0.3)   # 20px apart, < min_lane_width_px
    res, dbg = run([a, b])
    assert res.mode in ("left_only", "right_only") and res.boundary_count == 2
    assert any("[MERGE]" in line for line in dbg["log"])
    assert res.left_x == 140.0 or res.right_x == 140.0            # the higher-weight anchor wins


@pytest.mark.software
def test_anchors_at_exactly_min_lane_width_are_not_merged():
    a, b = cand(100), cand(160)                                   # exactly 60px apart
    res, dbg = run([a, b])
    assert res.mode == "two_boundary"
    assert not any("[MERGE]" in line for line in dbg["log"])


@pytest.mark.software
def test_anchors_too_far_apart_span_into_a_single_sided_read():
    a, b = cand(50, confidence=0.9, proximity=0.9), cand(500, confidence=0.4, proximity=0.4)
    res, dbg = run([a, b])
    assert res.mode in ("left_only", "right_only") and res.boundary_count == 2
    assert any("[SPAN]" in line for line in dbg["log"])
    assert res.left_x == 50.0 or res.right_x == 50.0              # the higher-weight anchor wins


@pytest.mark.software
def test_anchors_at_exactly_max_lane_width_are_not_a_span():
    a, b = cand(50), cand(450)                                    # exactly 400px apart
    res, dbg = run([a, b])
    assert res.mode == "two_boundary"
    assert not any("[SPAN]" in line for line in dbg["log"])


@pytest.mark.software
def test_left_only_projects_the_lane_center_to_the_right_of_the_boundary():
    res, dbg = run([cand(100)], cfg=TEST_CFG)
    assert res.mode == "left_only" and res.left_x == 100.0 and res.right_x is None
    implied_center = 100.0 + TEST_CFG.expected_half_lane_px
    assert res.offset == round((CENTER_X - implied_center) / CENTER_X, 4)
    assert res.lane_width_px is None


@pytest.mark.software
def test_right_only_projects_the_lane_center_to_the_left_of_the_boundary():
    res, _ = run([cand(220)], cfg=TEST_CFG)
    assert res.mode == "right_only" and res.right_x == 220.0 and res.left_x is None
    implied_center = 220.0 - TEST_CFG.expected_half_lane_px
    assert res.offset == round((CENTER_X - implied_center) / CENTER_X, 4)


@pytest.mark.software
def test_uncalibrated_single_boundary_emits_no_steering_signal():
    res, dbg = run([cand(100)], cfg=UNCALIBRATED)
    assert res.mode == "single_uncalibrated"
    assert res.offset == 0.0 and res.confidence == 0.0
    assert res.left_x is None and res.right_x is None
    assert any("[UNCALIBRATED]" in line for line in dbg["log"])
    assert res.boundary_count == 1                                # still reports what geometry saw


@pytest.mark.software
def test_uncalibrated_two_boundary_mode_is_unaffected():
    # expected_half_lane_px only gates the single-boundary path
    res, _ = run([cand(100), cand(200)], cfg=UNCALIBRATED)
    assert res.mode == "two_boundary"


@pytest.mark.software
def test_no_candidates_at_all_is_mode_none_with_an_empty_log():
    res, dbg = run([])
    assert res.mode == "none" and res.boundary_count == 0
    assert res.offset == 0.0 and res.confidence == 0.0
    assert dbg["raw_count"] == 0 and dbg["log"] == []


@pytest.mark.software
def test_candidates_present_but_none_usable_is_blind_not_none_silently():
    res, dbg = run([cand(100, confidence=0.01), cand(200, mean_intensity=1.0)])
    assert res.mode == "none" and res.boundary_count == 0
    assert dbg["raw_count"] == 2 and dbg["usable_count"] == 0
    assert any("[BLIND]" in line and "2 candidates" in line for line in dbg["log"])


@pytest.mark.software
def test_boundary_count_reports_usable_not_raw_detections():
    res, dbg = run([cand(100), cand(150, confidence=0.01), cand(200)])
    assert res.boundary_count == 2 and dbg["raw_count"] == 3


@pytest.mark.software
@pytest.mark.parametrize("cands", [[], [cand(100)], [cand(100), cand(200)],
                                   [cand(100, confidence=0.01)], [cand(-50), cand(400)]])
def test_contract_holds_across_every_mode(cands):
    roi = make_roi()
    res, dbg = compute_lane_offset(geo(cands), roi, TEST_CFG)
    assert_offset_contract(res, dbg, roi)


@pytest.mark.software
def test_identity_is_carried_not_rederived():
    res, dbg = run([cand(100)], frame_id=987654, ts=555)
    assert (res.frame_id, res.timestamp_ms) == (987654, 555) == (dbg["frame_id"], dbg["timestamp_ms"])


@pytest.mark.software
def test_deterministic():
    a, _ = run([cand(100), cand(200)])
    b, _ = run([cand(100), cand(200)])
    assert a == b


@pytest.mark.software
def test_result_and_anchor_are_frozen():
    res, _ = run([cand(100)])
    with pytest.raises(Exception):
        res.offset = 0.0
    anchor = lo._anchor(cand(100), TEST_CFG)
    with pytest.raises(Exception):
        anchor.weight = 0.0


@pytest.mark.software
def test_none_geometry_or_roi_is_rejected():
    with pytest.raises(ValueError, match="geometry"):
        compute_lane_offset(None, make_roi(), TEST_CFG)
    with pytest.raises(ValueError, match="roi"):
        compute_lane_offset(geo([]), None, TEST_CFG)


@pytest.mark.software
def test_mismatched_frame_stamps_are_rejected():
    mismatched_roi = make_roi(frame_id=99, ts=2)
    with pytest.raises(ValueError, match="different frames"):
        compute_lane_offset(geo([cand(100)], frame_id=1, ts=2), mismatched_roi, TEST_CFG)


def build_lane_scene(cx_frac, H=FRAME_H, W=FRAME_W):
    """A single vertical tape at cx_frac of the lane ROI's width, run through real geometry."""
    frame = np.full((H, W, 3), 30, np.uint8)
    probe = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig())
    lx, ly, lw, lh = probe.lane_rect
    tx = lx + int(cx_frac * lw)
    cv2.line(frame, (tx, ly + int(0.1 * lh)), (tx, ly + int(0.95 * lh)), (230, 230, 230), 8)
    roi = crop_rois(preprocess_frame(FrameData(frame, 21, 99)), ROIConfig())
    geometry, _, _ = run_geometry_stage(roi, GeometryConfig())
    return geometry, roi, tx - lx


@pytest.mark.software
def test_chain_a_single_real_boundary_yields_a_one_sided_read_near_its_true_position():
    geometry, roi, expected_x = build_lane_scene(cx_frac=0.5)
    assert len(geometry.lane_candidates) == 1
    res, dbg = compute_lane_offset(geometry, roi, TEST_CFG)
    assert_offset_contract(res, dbg, roi)
    assert res.mode in ("left_only", "right_only", "single_uncalibrated")
    got_x = res.left_x if res.left_x is not None else res.right_x
    if got_x is not None:
        assert abs(got_x - expected_x) <= 6       # 8 px tape, blurred by preprocess before Canny


@pytest.mark.software
def test_chain_two_real_boundaries_bracket_a_centered_robot():
    frame = np.full((FRAME_H, FRAME_W, 3), 30, np.uint8)
    probe = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig())
    lx, ly, lw, lh = probe.lane_rect
    for frac in (0.15, 0.85):
        tx = lx + int(frac * lw)
        cv2.line(frame, (tx, ly + int(0.1 * lh)), (tx, ly + int(0.95 * lh)), (230, 230, 230), 8)
    roi = crop_rois(preprocess_frame(FrameData(frame, 21, 99)), ROIConfig())
    geometry, _, _ = run_geometry_stage(roi, GeometryConfig())
    res, dbg = compute_lane_offset(geometry, roi, TEST_CFG)
    assert_offset_contract(res, dbg, roi)
    assert res.mode == "two_boundary"
    assert abs(res.offset) < 0.15                     # roughly centered between two symmetric lines


@pytest.mark.software
def test_every_recorded_frame_meets_the_offset_contract(dataset_frames):
    if not dataset_frames:
        pytest.skip("no recorded dataset in tests/data/frames (run: pytest --hardware --record)")
    for fd in dataset_frames:
        roi = crop_rois(preprocess_frame(fd))
        geometry, _, _ = run_geometry_stage(roi, GeometryConfig())
        try:
            res, dbg = compute_lane_offset(geometry, roi, TEST_CFG)
            assert_offset_contract(res, dbg, roi)
        except AssertionError as e:
            raise AssertionError(f"frame_id={fd.frame_id}: {e}") from e


def draw_anchor_overlay(lane_roi, dbg, result):
    """Lane ROI with the center line (cyan), each anchor (green if weight >= 0.5, else orange) and the result label."""
    vis = cv2.cvtColor(lane_roi, cv2.COLOR_GRAY2BGR) if lane_roi.ndim == 2 else lane_roi.copy()
    h = vis.shape[0]
    cx = int(vis.shape[1] / 2)
    cv2.line(vis, (cx, 0), (cx, h - 1), (255, 255, 0), 1)
    for x, weight in dbg["anchors"]:
        color = (0, 255, 0) if weight >= 0.5 else (0, 165, 255)
        cv2.line(vis, (int(x), 0), (int(x), h - 1), color, 1)
    label = f"{result.mode} off={result.offset:+.3f} conf={result.confidence:.2f}"
    cv2.putText(vis, label, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


@pytest.mark.hardware
def test_lane_offset_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    geo_cfg, lo_cfg = GeometryConfig(), LaneOffsetConfig()

    rows, samples = [], {}
    mode_counts = {m: 0 for m in MODES}
    for i, fd in enumerate(frames(n)):
        roi = crop_rois(preprocess_frame(fd))
        geometry, _, _ = run_geometry_stage(roi, geo_cfg)          # upstream, not timed

        t0 = time.perf_counter_ns()
        res, dbg = compute_lane_offset(geometry, roi, lo_cfg)
        stage_ms = (time.perf_counter_ns() - t0) / 1e6

        assert_offset_contract(res, dbg, roi)                      # outside the timing window
        mode_counts[res.mode] += 1
        rows.append((fd.frame_id, fd.timestamp_ms, round(stage_ms, 3), res.mode, res.offset,
                     res.left_x, res.right_x, res.lane_width_px, res.confidence,
                     res.boundary_count, dbg["raw_count"]))
        if i in (0, n // 2, n - 1):
            samples[fd.frame_id] = (roi, dbg, res)

    if not rows:
        pytest.skip("no frames delivered")

    stage = [r[2] for r in rows]
    offsets = [r[4] for r in rows]
    artifacts.json("config.json", asdict(lo_cfg))
    artifacts.csv("lane_offset_timing.csv",
                  ["frame_id", "timestamp_ms", "stage_ms", "mode", "offset", "left_x", "right_x",
                   "lane_width_px", "confidence", "boundary_count", "raw_count"], rows)
    artifacts.json("summary.json", {
        "stage_ms": summarize(stage), "offset": summarize(offsets),
        "mode_counts": mode_counts, "frames_blind": mode_counts["none"],
    })
    artifacts.histogram("stage_ms_hist.png", stage, "compute_lane_offset latency", "ms")
    artifacts.histogram("offset_hist.png", offsets, "Lane offset distribution", "offset")

    for fid, (roi, dbg, res) in samples.items():
        artifacts.image(f"{fid:06d}_anchors.png", draw_anchor_overlay(roi.lane_roi, dbg, res))
        artifacts.json(f"{fid:06d}_debug.json", dbg)