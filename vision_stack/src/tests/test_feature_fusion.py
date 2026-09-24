"""
test_feature_fusion.py  --  src/perception/feature_fusion.py

fuse_detections consumes GeometryBranchResult plus a duck-typed list of color
candidates (label, bbox, confidence, frame_id), so software tests build both
by hand. Color candidates never need to be TrafficLightCandidate instances --
a test fake with the right attributes is enough, and one test proves that.

Coordinate spaces stay ROI-local per the module docstring: position and
bounding_box are in the coordinates of source_roi, not the frame. Chained
tests convert to frame coordinates explicitly (position + source_rect origin)
rather than assuming fusion did it.

--software  Contract, ordering, conflict-resolution and known-answer tests.
--hardware  Times fuse_detections per frame (live or --replay), chaining
            geometry and color, and writes CSV, per-class counts, and overlays.
"""
import math
import time
from dataclasses import FrozenInstanceError, asdict, replace

import cv2
import numpy as np
import pytest

from src.capture.camera import FrameData
from src.perception import feature_fusion as ff
from src.perception.color_branch import (
    BlobFilter, ColorConfig, ColorRange, HSVRanges, load_color_config, run_color_stage,
)
from src.perception.feature_fusion import (
    DetectionObject, FusionResult, draw_fusion_overlay, fuse_detections,
)
from src.perception.geometry import GeometryBranchResult, GeometryConfig, LaneCandidate, SignCandidate, run_geometry_stage
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import ROIConfig, ROICropResult, crop_rois
from src.params import FRAME_H, FRAME_W
from src.tests.artifacts import summarize

# Explicit color tuning for the chained test, independent of the scaffold and any calibration
TEST_HSV = HSVRanges(
    red_low=ColorRange((0, 120, 120), (10, 255, 255)),
    red_high=ColorRange((170, 120, 120), (180, 255, 255)),
    yellow=ColorRange((20, 120, 120), (35, 255, 255)),
    green=ColorRange((40, 120, 120), (80, 255, 255)))
TEST_BLOB = BlobFilter(min_area=50.0, max_area=5000.0, min_aspect=0.3, max_aspect=3.0, ref_area=800.0)
TEST_COLOR_CFG = ColorConfig(TEST_HSV, TEST_BLOB)

TYPES = ("traffic_light", "lane_boundary", "stop_sign")


class FakeTrafficCandidate:
    """Minimal duck type fuse_detections actually reads: label, bbox, confidence, frame_id."""
    def __init__(self, label, bbox, confidence, frame_id=1, timestamp_ms=2):
        self.label, self.bbox, self.confidence = label, bbox, confidence
        self.frame_id, self.timestamp_ms = frame_id, timestamp_ms


def lane(fx, confidence=0.9, bbox=(10, 10, 8, 8), frame_id=1, ts=2):
    """Hand-built LaneCandidate with foot_x = fx; no contour, since fusion never reads it."""
    return LaneCandidate("lane_boundary", bbox, None, confidence, frame_id, ts, 0.5, 3.0, 20.0, 200.0, fx)


def sign(confidence=0.9, vertex_count=8, bbox=(30, 30, 20, 20), area=300.0, solidity=0.95, frame_id=1, ts=2):
    """Hand-built SignCandidate; no contour, since fusion never reads it."""
    return SignCandidate("stop_sign", bbox, None, vertex_count, confidence, frame_id, ts, area, solidity)


def make_roi(frame_id=1, ts=2, lane_rect=(0, 0, 300, 50), traffic_rect=(10, 10, 200, 150), sign_rect=(300, 0, 150, 150)):
    """ROICropResult carrying only rects and a stamp; the 1x1 ROI arrays are placeholders fusion never reads."""
    return ROICropResult(np.zeros((1, 1), np.uint8), np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8),
                         lane_rect, traffic_rect, sign_rect, frame_id, ts, (FRAME_H, FRAME_W))


def geo(lanes=(), signs=(), frame_id=1, ts=2):
    """GeometryBranchResult from candidate lists."""
    return GeometryBranchResult(list(lanes), list(signs), frame_id, ts)


def run(lanes=(), signs=(), tls=(), roi=None, frame_id=1, ts=2):
    """fuse_detections on hand-built inputs sharing one frame stamp."""
    roi = roi or make_roi(frame_id=frame_id, ts=ts)
    return fuse_detections(geo(lanes, signs, frame_id, ts), list(tls), roi)


def by_type(detections, t):
    """Detections of one class, in output order."""
    return [d for d in detections if d.type == t]


def assert_fusion_contract(result, dbg, roi):
    """Everything FusionResult, DetectionObject and the debug summary document."""
    assert isinstance(result, FusionResult)
    assert (result.frame_id, result.timestamp_ms) == (roi.frame_id, roi.timestamp_ms) == \
           (dbg["frame_id"], dbg["timestamp_ms"])
    assert dbg["total"] == len(result.detections)
    counted = {}
    for d in result.detections:
        assert isinstance(d, DetectionObject) and d.type in TYPES
        assert 0.0 <= d.confidence <= 1.0
        assert (d.frame_id, d.timestamp) == (roi.frame_id, roi.timestamp_ms)
        x, y, w, h = d.bounding_box
        assert d.position["x"] == round(x + w / 2.0, 2) or w <= 0
        assert d.position["y"] == round(y + h / 2.0, 2) or h <= 0
        assert d.source_roi in ("lane", "traffic", "sign")
        expect_rect = {"lane": roi.lane_rect, "traffic": roi.traffic_rect, "sign": roi.sign_rect}[d.source_roi]
        assert d.source_rect == expect_rect
        counted[d.type] = counted.get(d.type, 0) + 1
    assert counted == dbg["counts"]
    assert len(by_type(result.detections, "traffic_light")) <= 1
    assert len(by_type(result.detections, "stop_sign")) <= 1
    lanes_out = by_type(result.detections, "lane_boundary")
    assert [d.confidence for d in lanes_out] == sorted((d.confidence for d in lanes_out), reverse=True)
    assert dbg["discarded"] == sum(1 for line in dbg["log"] if "[DISCARD]" in line)
    assert dbg["suppressed"] == sum(1 for line in dbg["log"] if "[SUPPRESSED]" in line)


@pytest.mark.software
@pytest.mark.parametrize("bbox, expect", [
    ((10, 20, 8, 4), {"x": 14.0, "y": 22.0}),
    ((10, 20, 0, 4), {"x": 10.0, "y": 22.0}),      # zero width: guard falls back to x
    ((10, 20, 8, 0), {"x": 14.0, "y": 20.0}),      # zero height: guard falls back to y
    ((10, 20, 0, 0), {"x": 10.0, "y": 20.0}),
])
def test_centroid_of_documented_zero_size_cases(bbox, expect):
    assert ff._centroid(bbox) == expect


@pytest.mark.software
def test_centroid_rounds_to_two_places():
    assert ff._centroid((0, 0, 3, 3)) == {"x": 1.5, "y": 1.5}


@pytest.mark.software
@pytest.mark.parametrize("c, ok", [(0.0, True), (1.0, True), (0.5, True),
                                   (-0.0001, False), (1.0001, False), (float("nan"), False)])
def test_valid_confidence_boundaries(c, ok):
    assert ff._valid_confidence(c) is ok


@pytest.mark.software
def test_best_candidate_of_empty_list_is_none_and_logs_nothing():
    log = []
    assert ff._best_candidate([], log, "traffic_light") is None and log == []


@pytest.mark.software
def test_best_candidate_discards_invalid_and_logs_the_class_name():
    log = []
    valid = FakeTrafficCandidate("green", (0, 0, 1, 1), 0.5)
    out = ff._best_candidate([FakeTrafficCandidate("red", (0, 0, 1, 1), float("nan")), valid], log, "traffic_light")
    assert out is valid
    assert len(log) == 1 and "[DISCARD]" in log[0] and "traffic_light" in log[0]


@pytest.mark.software
def test_best_candidate_all_invalid_is_none_with_every_one_logged():
    log = []
    out = ff._best_candidate([FakeTrafficCandidate("r", (0, 0, 1, 1), -1),
                              FakeTrafficCandidate("g", (0, 0, 1, 1), float("nan"))], log, "stop_sign")
    assert out is None and len(log) == 2


@pytest.mark.software
def test_best_candidate_breaks_ties_by_keeping_the_first():
    a, b = FakeTrafficCandidate("red", (0, 0, 1, 1), 0.5), FakeTrafficCandidate("green", (0, 0, 1, 1), 0.5)
    assert ff._best_candidate([a, b], [], "traffic_light") is a
    assert ff._best_candidate([b, a], [], "traffic_light") is b


@pytest.mark.software
def test_color_branch_module_is_never_imported():
    # Fusion must load and fuse lane/sign detections without the color branch
    import sys
    assert "src.perception.color_branch" not in getattr(ff, "__dict__", {}).values()
    assert not hasattr(ff, "color_branch")


@pytest.mark.software
def test_a_plain_duck_typed_object_is_accepted_as_a_color_candidate():
    (result, _) = run(tls=[FakeTrafficCandidate("red", (5, 5, 10, 10), 0.8)])
    assert [d.type for d in result.detections] == ["traffic_light"]
    assert result.detections[0].label_detail == "red"


@pytest.mark.software
def test_detections_are_ordered_traffic_then_lane_desc_then_sign():
    lanes = [lane(10, confidence=0.3), lane(200, confidence=0.8), lane(150, confidence=0.5)]
    tls = [FakeTrafficCandidate("red", (0, 0, 5, 5), 0.6), FakeTrafficCandidate("green", (5, 5, 5, 5), 0.9)]
    signs = [sign(confidence=0.4), sign(confidence=0.7)]
    roi = make_roi()
    result, dbg = fuse_detections(geo(lanes, signs), tls, roi)
    assert [d.type for d in result.detections] == ["traffic_light", "lane_boundary", "lane_boundary", "lane_boundary", "stop_sign"]
    assert [d.confidence for d in by_type(result.detections, "lane_boundary")] == [0.8, 0.5, 0.3]
    assert result.detections[0].confidence == 0.9 and result.detections[-1].confidence == 0.7
    assert_fusion_contract(result, dbg, roi)


@pytest.mark.software
def test_each_detection_carries_its_own_roi_name_and_frame_rect():
    roi = make_roi(lane_rect=(1, 2, 3, 4), traffic_rect=(5, 6, 7, 8), sign_rect=(9, 10, 11, 12))
    result, _ = fuse_detections(geo([lane(10)], [sign()]), [FakeTrafficCandidate("g", (0, 0, 1, 1), 0.5)], roi)
    rects = {d.type: (d.source_roi, d.source_rect) for d in result.detections}
    assert rects == {
        "traffic_light": ("traffic", (5, 6, 7, 8)),
        "lane_boundary": ("lane", (1, 2, 3, 4)),
        "stop_sign": ("sign", (9, 10, 11, 12)),
    }


@pytest.mark.software
def test_empty_input_gives_no_detections_and_an_empty_debug_summary():
    result, dbg = run()
    assert result.detections == [] and dbg == {
        "frame_id": 1, "timestamp_ms": 2, "counts": {}, "total": 0,
        "discarded": 0, "suppressed": 0, "log": [],
    }


@pytest.mark.software
def test_traffic_light_keeps_only_the_winner_and_suppresses_the_rest():
    tls = [FakeTrafficCandidate("red", (0, 0, 5, 5), 0.6), FakeTrafficCandidate("green", (5, 5, 5, 5), 0.9),
          FakeTrafficCandidate("yellow", (1, 1, 5, 5), 0.9)]        # a genuine tie with the winner
    result, dbg = run(tls=tls)
    winners = by_type(result.detections, "traffic_light")
    assert len(winners) == 1 and winners[0].confidence == 0.9 and winners[0].label_detail == "green"
    assert dbg["suppressed"] == 2 and all("traffic_light" in l for l in dbg["log"] if "[SUPPRESSED]" in l)


@pytest.mark.software
def test_stop_sign_keeps_only_the_winner_and_suppresses_the_rest():
    signs = [sign(confidence=0.4, vertex_count=7), sign(confidence=0.8, vertex_count=8), sign(confidence=0.2, vertex_count=9)]
    result, dbg = run(signs=signs)
    winners = by_type(result.detections, "stop_sign")
    assert len(winners) == 1 and winners[0].confidence == 0.8
    assert dbg["suppressed"] == 2
    assert any("v=8" in l and "v=7" in l for l in dbg["log"] if "[SUPPRESSED]" in l) or \
          any("v=8" in l and "v=9" in l for l in dbg["log"] if "[SUPPRESSED]" in l)


@pytest.mark.software
def test_lane_boundary_forwards_every_valid_candidate_none_are_suppressed():
    lanes = [lane(10, confidence=0.9), lane(20, confidence=0.9), lane(30, confidence=0.1)]
    result, dbg = run(lanes=lanes)
    assert len(by_type(result.detections, "lane_boundary")) == 3
    assert dbg["suppressed"] == 0 and not any("SUPPRESSED" in l and "lane" in l.lower() for l in dbg["log"])


@pytest.mark.software
def test_lane_boundary_stable_sort_preserves_input_order_among_equal_confidence():
    la = lane(10, confidence=0.5, bbox=(1, 1, 1, 1))
    lb = lane(20, confidence=0.5, bbox=(2, 2, 1, 1))
    lc = lane(30, confidence=0.9, bbox=(3, 3, 1, 1))
    result, _ = run(lanes=[la, lb, lc])
    assert [d.bounding_box for d in by_type(result.detections, "lane_boundary")] == [(3, 3, 1, 1), (1, 1, 1, 1), (2, 2, 1, 1)]


@pytest.mark.software
@pytest.mark.parametrize("cls, make", [
    ("lane_boundary", lambda c: lane(10, confidence=c)),
    ("stop_sign", lambda c: sign(confidence=c)),
    ("traffic_light", lambda c: FakeTrafficCandidate("r", (0, 0, 1, 1), c)),
])
def test_out_of_range_confidence_is_discarded_and_logged_for_every_class(cls, make):
    kwargs = {"lanes": [make(1.5)]} if cls == "lane_boundary" else \
             {"signs": [make(-0.1)]} if cls == "stop_sign" else {"tls": [make(float("nan"))]}
    result, dbg = run(**kwargs)
    assert by_type(result.detections, cls) == []
    assert dbg["discarded"] == 1 and any("[DISCARD]" in l and cls in l for l in dbg["log"])


@pytest.mark.software
def test_a_class_with_one_valid_and_one_invalid_forwards_only_the_valid_one():
    result, dbg = run(lanes=[lane(10, confidence=0.7), lane(20, confidence=2.0)])
    assert [d.confidence for d in by_type(result.detections, "lane_boundary")] == [0.7]
    assert dbg["discarded"] == 1


@pytest.mark.software
def test_stamp_comes_from_roi_not_from_any_candidates_clock():
    # A branch sampling its own clock must not split one capture's detections
    roi = make_roi(frame_id=1, ts=2)
    spoofed_tl = FakeTrafficCandidate("red", (0, 0, 1, 1), 0.9, frame_id=555, timestamp_ms=777)
    result, dbg = fuse_detections(geo([], []), [spoofed_tl], roi)
    assert (result.detections[0].frame_id, result.detections[0].timestamp) == (1, 2)
    assert (dbg["frame_id"], dbg["timestamp_ms"]) == (1, 2)


@pytest.mark.software
def test_mismatched_geometry_and_roi_stamps_are_rejected():
    with pytest.raises(ValueError, match="different frames"):
        fuse_detections(geo([], [], frame_id=99), [], make_roi(frame_id=1))


@pytest.mark.software
def test_none_geometry_or_roi_is_rejected():
    with pytest.raises(ValueError, match="geometry"):
        fuse_detections(None, [], make_roi())
    with pytest.raises(ValueError, match="roi"):
        fuse_detections(geo([], []), [], None)


@pytest.mark.software
@pytest.mark.parametrize("lanes, signs, tls", [
    ([], [], []),
    ([lane(10)], [], []),
    ([], [sign()], []),
    ([], [], [FakeTrafficCandidate("g", (0, 0, 5, 5), 0.8)]),
    ([lane(10, confidence=1.5)], [sign(confidence=-1)], [FakeTrafficCandidate("g", (0, 0, 5, 5), float("nan"))]),
    ([lane(x) for x in range(0, 200, 20)], [sign(confidence=0.3), sign(confidence=0.9)],
     [FakeTrafficCandidate("r", (0, 0, 5, 5), 0.4), FakeTrafficCandidate("g", (5, 5, 5, 5), 0.6)]),
])
def test_contract_holds_across_realistic_input_mixes(lanes, signs, tls):
    roi = make_roi()
    result, dbg = fuse_detections(geo(lanes, signs), tls, roi)
    assert_fusion_contract(result, dbg, roi)


@pytest.mark.software
def test_fusion_result_and_detection_object_are_frozen():
    result, _ = run(lanes=[lane(10)])
    with pytest.raises(FrozenInstanceError):
        result.frame_id = 0
    with pytest.raises(FrozenInstanceError):
        result.detections[0].confidence = 0.0


@pytest.mark.software
def test_deterministic():
    a, _ = run(lanes=[lane(10, confidence=0.5), lane(20, confidence=0.9)], signs=[sign()],
              tls=[FakeTrafficCandidate("g", (0, 0, 5, 5), 0.7)])
    b, _ = run(lanes=[lane(10, confidence=0.5), lane(20, confidence=0.9)], signs=[sign()],
              tls=[FakeTrafficCandidate("g", (0, 0, 5, 5), 0.7)])
    assert a == b


@pytest.mark.software
def test_overlay_returns_a_copy_and_leaves_the_canvas_alone():
    canvas = np.zeros((150, 200, 3), np.uint8)
    before = canvas.copy()
    result, _ = run(lanes=[lane(10)])
    vis = draw_fusion_overlay(canvas, result.detections)
    assert np.array_equal(canvas, before) and vis is not canvas and not np.shares_memory(vis, canvas)


@pytest.mark.software
def test_overlay_source_roi_filter_draws_only_that_rois_detections():
    result, _ = run(lanes=[lane(10, bbox=(20, 20, 10, 10))],
                    tls=[FakeTrafficCandidate("g", (60, 60, 10, 10), 0.8)])
    canvas = np.zeros((150, 200, 3), np.uint8)
    lane_only = draw_fusion_overlay(canvas, result.detections, source_roi="lane")
    traffic_only = draw_fusion_overlay(canvas, result.detections, source_roi="traffic")
    assert tuple(lane_only[25, 25]) == ff._TYPE_COLORS["lane_boundary"]
    assert tuple(lane_only[65, 65]) == (0, 0, 0)                    # traffic box not drawn here
    assert tuple(traffic_only[65, 65]) == ff._TYPE_COLORS["traffic_light"]
    assert tuple(traffic_only[25, 25]) == (0, 0, 0)


@pytest.mark.software
def test_overlay_unknown_type_falls_back_to_a_neutral_color():
    fake = DetectionObject("mystery_type", "x", 0.5, {"x": 25.0, "y": 25.0}, (20, 20, 10, 10),
                           "lane", (0, 0, 100, 100), 1, 2)
    vis = draw_fusion_overlay(np.zeros((50, 100, 3), np.uint8), [fake])
    assert tuple(vis[20, 20]) == (200, 200, 200)


@pytest.mark.software
def test_overlay_title_is_only_drawn_when_given():
    result, _ = run(lanes=[lane(10, bbox=(20, 20, 10, 10))])
    plain = draw_fusion_overlay(np.zeros((50, 100, 3), np.uint8), result.detections)
    titled = draw_fusion_overlay(np.zeros((50, 100, 3), np.uint8), result.detections, title="frame 7")
    assert not np.array_equal(plain, titled)


def build_fusion_scene(frame_id=11, ts=222, H=FRAME_H, W=FRAME_W):
    """Frame with a tape, an octagon and a green light placed from the default rects; returns (roi, expected frame coords)."""
    frame = np.full((H, W, 3), 30, np.uint8)
    probe = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig())
    lx, ly, lw, lh = probe.lane_rect
    tx_lane = lx + int(0.3 * lw)
    cv2.line(frame, (tx_lane, ly + int(0.1 * lh)), (tx_lane, ly + int(0.9 * lh)), (230, 230, 230), 8)
    sx, sy, sw, sh = probe.sign_rect
    ox, oy, r = sx + sw // 2, sy + sh // 2, int(0.3 * min(sw, sh))
    pts = np.array([(ox + r * np.cos(np.pi / 8 + 2 * np.pi * k / 8), oy + r * np.sin(np.pi / 8 + 2 * np.pi * k / 8))
                    for k in range(8)], np.int32)
    cv2.fillPoly(frame, [pts], (230, 230, 230))
    tx, ty, tw, th = probe.traffic_rect
    gcx, gcy = tx + int(0.5 * tw), ty + int(0.5 * th)
    cv2.circle(frame, (gcx, gcy), int(0.1 * min(tw, th)), (0, 255, 0), -1)
    roi = crop_rois(preprocess_frame(FrameData(frame, frame_id, ts)), ROIConfig())
    return roi, {"lane_x": tx_lane, "sign_center": (ox, oy), "traffic_center": (gcx, gcy)}


@pytest.mark.software
def test_chain_fuses_real_geometry_and_color_output_at_frame_coordinates():
    roi, expect = build_fusion_scene()
    geometry, _, _ = run_geometry_stage(roi, GeometryConfig())
    tls, _ = run_color_stage(roi, TEST_COLOR_CFG)
    result, dbg = fuse_detections(geometry, tls, roi)
    assert_fusion_contract(result, dbg, roi)

    lane_dets = by_type(result.detections, "lane_boundary")
    assert len(lane_dets) == 1
    fx = lane_dets[0].position["x"] + lane_dets[0].source_rect[0]
    assert abs(fx - expect["lane_x"]) <= 8           # 8 px tape, blurred by preprocess before Canny

    sign_dets = by_type(result.detections, "stop_sign")
    assert len(sign_dets) == 1
    fx = sign_dets[0].position["x"] + sign_dets[0].source_rect[0]
    fy = sign_dets[0].position["y"] + sign_dets[0].source_rect[1]
    assert abs(fx - expect["sign_center"][0]) <= 6 and abs(fy - expect["sign_center"][1]) <= 6

    tl_dets = by_type(result.detections, "traffic_light")
    assert len(tl_dets) == 1 and tl_dets[0].label_detail == "green"
    fx = tl_dets[0].position["x"] + tl_dets[0].source_rect[0]
    fy = tl_dets[0].position["y"] + tl_dets[0].source_rect[1]
    assert abs(fx - expect["traffic_center"][0]) <= 4 and abs(fy - expect["traffic_center"][1]) <= 4


@pytest.mark.software
def test_every_recorded_frame_meets_the_fusion_contract(dataset_frames):
    if not dataset_frames:
        pytest.skip("no recorded dataset in tests/data/frames (run: pytest --hardware --record)")
    for fd in dataset_frames:
        roi = crop_rois(preprocess_frame(fd))
        geometry, _, _ = run_geometry_stage(roi, GeometryConfig())
        tls, _ = run_color_stage(roi, ColorConfig(HSVRanges(), BlobFilter()))
        try:
            result, dbg = fuse_detections(geometry, tls, roi)
            assert_fusion_contract(result, dbg, roi)
        except AssertionError as e:
            raise AssertionError(f"frame_id={fd.frame_id}: {e}") from e


def _hardware_color_config():
    """Calibrated color config if the calibration file exists, else the scaffold; returns (config, source)."""
    from pathlib import Path
    calib = Path(__file__).resolve().parents[2] / "calibration" / "hsv_ranges.json"
    if calib.exists():
        return load_color_config(str(calib)), str(calib)
    return ColorConfig(HSVRanges(), BlobFilter()), "scaffold (uncalibrated)"


@pytest.mark.hardware
def test_fusion_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    geo_cfg = GeometryConfig()
    color_cfg, color_source = _hardware_color_config()

    rows, samples = [], {}
    type_totals = {t: 0 for t in TYPES}
    for i, fd in enumerate(frames(n)):
        roi = crop_rois(preprocess_frame(fd))
        geometry, _, _ = run_geometry_stage(roi, geo_cfg)
        tls, _ = run_color_stage(roi, color_cfg)     # upstream, not timed

        t0 = time.perf_counter_ns()
        result, dbg = fuse_detections(geometry, tls, roi)
        stage_ms = (time.perf_counter_ns() - t0) / 1e6

        assert_fusion_contract(result, dbg, roi)                      # outside the timing window
        for t in TYPES:
            type_totals[t] += dbg["counts"].get(t, 0)
        rows.append((fd.frame_id, fd.timestamp_ms, round(stage_ms, 4), dbg["total"],
                     *(dbg["counts"].get(t, 0) for t in TYPES), dbg["discarded"], dbg["suppressed"]))
        if i in (0, n // 2, n - 1):
            samples[fd.frame_id] = (roi, result, dbg)

    if not rows:
        pytest.skip("no frames delivered")

    stage = [r[2] for r in rows]
    artifacts.json("config.json", {"color_source": color_source, "geometry": asdict(geo_cfg)})
    artifacts.csv("fusion_timing.csv",
                  ["frame_id", "timestamp_ms", "stage_ms", "total"] + [f"n_{t}" for t in TYPES]
                  + ["discarded", "suppressed"], rows)
    artifacts.json("summary.json", {
        "stage_ms": summarize(stage), "type_totals": type_totals,
        "frames_with_any_detection": sum(1 for r in rows if r[3] > 0),
        "total_discarded": sum(r[-2] for r in rows), "total_suppressed": sum(r[-1] for r in rows),
    })
    artifacts.histogram("stage_ms_hist.png", stage, "fuse_detections latency", "ms")

    for fid, (roi, result, dbg) in samples.items():
        artifacts.image(f"{fid:06d}_lane_overlay.png",
                        draw_fusion_overlay(roi.lane_roi, result.detections, source_roi="lane"))
        artifacts.image(f"{fid:06d}_traffic_overlay.png",
                        draw_fusion_overlay(roi.traffic_roi, result.detections, source_roi="traffic"))
        artifacts.image(f"{fid:06d}_sign_overlay.png",
                        draw_fusion_overlay(roi.sign_roi, result.detections, source_roi="sign"))
        artifacts.json(f"{fid:06d}_debug.json", dbg)