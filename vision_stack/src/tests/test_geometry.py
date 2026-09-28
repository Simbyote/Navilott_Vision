"""
test_geometry.py  --  src/perception/geometry.py

Detection tests use synthetic ROIs with known ground truth (white tape on a dark
mat, a filled octagon) and their own explicit filter/Canny parameters, so
retuning the shipped defaults never breaks them. Invariant tests (candidate
bounds, reject-count bookkeeping) run on noise-and-shapes ROIs and on real data.
Coordinates are ROI-relative; the chained tests draw shapes at positions derived
from crop_rois()'s rects, so they hold for any ROI bounds.

--software  Contract, gate-wiring, known-answer and chained tests. No camera.
--hardware  Times run_geometry_stage per frame (live or --replay), logs reject
            counts per gate, and writes edge maps / overlays / sign trace.
"""
import time
from types import SimpleNamespace
from dataclasses import asdict, replace

import cv2
import numpy as np
import pytest

from src.capture.camera import FrameData
from src.perception import geometry as geo
from src.perception.geometry import (
    CannyParams, GeometryBranchResult, GeometryConfig, LaneCandidate,
    LaneContourFilter, SignContourFilter, StopLineFilter,
    contour_foot_x, extract_lane_candidates, extract_sign_candidates,
    extract_stop_line_candidates, find_lane_candidates, run_geometry_branch, run_geometry_stage,
)
from src.perception.preprocess import PreprocessResult, preprocess_frame
from src.perception.roi_crop import ROIBounds, ROIConfig, crop_rois
from src.params import FRAME_H, FRAME_W, STOP_LINE
from src.tests.artifacts import summarize

# Explicit rather than the shipped defaults, so retuning never breaks detection tests
TEST_CANNY = CannyParams(threshold1=80.0, threshold2=200.0, aperture_size=3, close_kernel=(9, 3))
# The horizontal-line filter is off here so the contour gates are tested alone;
# it has its own tests below, with TEST_LANE_FILTERED
TEST_LANE = LaneContourFilter(min_area=1, max_area=1e6, min_aspect=0.0, max_aspect=1000.0,
                              max_roi_span=1.0, min_intensity=120.0, ref_length=0.25,
                              ref_width=30.0, horizontal_edge_deg=None)
TEST_LANE_FILTERED = replace(TEST_LANE, horizontal_edge_deg=20.0, horizontal_min_run_px=46.0,
                             horizontal_band_px=3.0)
TEST_SIGN = SignContourFilter(min_area=200.0, max_area=30000.0, min_vertices=8, max_vertices=10,
                              min_solidity=0.80, epsilon_factor=0.03, ref_area=5000.0)
TEST_STOP = StopLineFilter(max_tilt_deg=20.0, min_length_px=60.0, min_thickness_px=3.0,
                           max_thickness_px=40.0, min_intensity=130.0, min_edge_overlap=0.5,
                           close_kernel=(15, 1), ref_length_px=200.0)
TEST_CFG = GeometryConfig(canny=TEST_CANNY, lane=TEST_LANE, sign=TEST_SIGN, stop_line=TEST_STOP)

LANE_SHAPE = (108, 432)       # synthetic lane ROI, sized so the drawn scenes fit; any size works
SIGN_SHAPE = (198, 240)       # synthetic sign ROI, same
BG, FG = 30, 230              # dark mat, white tape

LANE_BUCKETS = ("area", "degenerate", "too_few_pts", "aspect", "w_span", "h_span", "intensity", "accepted")
SIGN_BUCKETS = ("area", "vertices", "hull", "solidity", "accepted")
STOP_BUCKETS = ("short", "tilt", "unpaired", "intensity", "accepted")
TRACE_KEYS = {"bbox", "gate", "area", "vertices", "solidity", "confidence", "poly"}


def blank(shape, value=BG):
    """Featureless ROI at one gray level."""
    return np.full(shape, value, np.uint8)


def slanted_tape(shape=LANE_SHAPE, cx=200, cy=80, length=200, thickness=8, angle_deg=3.0):
    """Filled rotated rectangle. A tiny slant keeps the outline from collapsing to 4 corners."""
    img = blank(shape)
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    nx, ny = -dy, dx
    L, T = length / 2, thickness / 2
    pts = [(cx - L*dx - T*nx, cy - L*dy - T*ny), (cx + L*dx - T*nx, cy + L*dy - T*ny),
           (cx + L*dx + T*nx, cy + L*dy + T*ny), (cx - L*dx + T*nx, cy - L*dy + T*ny)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], FG)
    return img


def line_tape(p0, p1, shape=LANE_SHAPE, thickness=8):
    """One straight tape stroke from p0 to p1 on a dark mat."""
    img = blank(shape)
    cv2.line(img, p0, p1, FG, thickness)
    return img


def regular_polygon(n, radius=40, center=(120, 100), shape=SIGN_SHAPE, rot=np.pi / 8):
    """Filled regular n-gon on a dark mat; rot=pi/8 gives an octagon flat on top, like a stop sign."""
    img = blank(shape)
    pts = np.array([(center[0] + radius*np.cos(rot + 2*np.pi*k/n),
                     center[1] + radius*np.sin(rot + 2*np.pi*k/n)) for k in range(n)], np.int32)
    cv2.fillPoly(img, [pts], FG)
    return img


def cont(pts):
    """Point list as an OpenCV (N, 1, 2) int32 contour."""
    return np.array(pts, np.int32).reshape(-1, 1, 2)


def octagon_contour(radius=40, center=(120, 100)):
    """Regular octagon contour, fed straight to the gates without Canny."""
    return cont([(center[0] + radius*np.cos(np.pi/8 + 2*np.pi*k/8),
                  center[1] + radius*np.sin(np.pi/8 + 2*np.pi*k/8)) for k in range(8)])


def star_contour(center=(120, 100)):
    """8-vertex concave star: a sign-like vertex count with low solidity."""
    return cont([(center[0] + (40 if k % 2 == 0 else 15)*np.cos(2*np.pi*k/8),
                  center[1] + (40 if k % 2 == 0 else 15)*np.sin(2*np.pi*k/8)) for k in range(8)])


SQUARE = cont([(0, 0), (60, 0), (60, 60), (0, 60)])
TINY = cont([(0, 0), (5, 0), (5, 5), (0, 5)])
SLIVER = cont([(0, 0), (100, 0), (100, 1), (0, 1)])    # tiny area, sign-sized bbox


def lane_cand(x0, x1, y=20, h=3, intensity=200.0, length=None):
    """Hand-built horizontal LaneCandidate (a 4-corner contour) for merge tests."""
    return LaneCandidate(
        label="lane_boundary", bbox=(x0, y, x1 - x0 + 1, h + 1),
        contour=cont([(x0, y), (x1, y), (x1, y + h), (x0, y + h)]),
        confidence=0.5, frame_id=1, timestamp_ms=2, proximity=0.5, width_px=3.0,
        length_px=float(x1 - x0) if length is None else length, mean_intensity=intensity)


def noisy_scene(seed, shape):
    """Noise plus random bright shapes: many contours, deterministic per seed."""
    rng = np.random.default_rng(seed)
    h, w = shape
    img = rng.integers(0, 60, shape, dtype=np.uint8)
    for _ in range(20):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            cv2.line(img, (x, y), (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(150, 256)), int(rng.integers(1, 9)))
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(3, 80)), y + int(rng.integers(3, 40))), int(rng.integers(150, 256)), -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(3, 30)), int(rng.integers(150, 256)), -1)
    return img


def assert_lane_candidates_ok(cands, shape, frame_id, ts):
    """Every documented LaneCandidate field is in range and inside the ROI."""
    h, w = shape[:2]
    for c in cands:
        x, y, bw, bh = c.bbox
        assert isinstance(c, LaneCandidate) and c.label == "lane_boundary"
        assert x >= 0 and y >= 0 and bw >= 1 and bh >= 1 and x + bw <= w and y + bh <= h, f"bbox {c.bbox} outside {shape}"
        assert 0.0 <= c.confidence <= 1.0 and 0.0 <= c.proximity <= 1.0
        assert c.length_px >= c.width_px >= 0.0
        assert 0.0 <= c.mean_intensity <= 255.0
        assert 0.0 <= c.foot_x <= w, f"foot_x {c.foot_x} (-1.0 means never computed)"
        assert (c.frame_id, c.timestamp_ms) == (frame_id, ts)
        assert c.contour.ndim == 3 and c.contour.shape[1:] == (1, 2)


def assert_sign_candidates_ok(cands, shape, frame_id, ts, flt=None):
    """Every documented SignCandidate field is in range; with flt, also within its gates."""
    h, w = shape[:2]
    for c in cands:
        x, y, bw, bh = c.bbox
        assert c.label == "stop_sign"
        assert x >= 0 and y >= 0 and bw >= 1 and bh >= 1 and x + bw <= w and y + bh <= h, f"bbox {c.bbox} outside {shape}"
        assert 0.0 <= c.confidence <= 1.0
        assert c.vertex_count == len(c.contour)
        assert c.area > 0.0 and 0.0 < c.solidity <= 1.0 + 1e-6            # float slack on area / hull area
        assert (c.frame_id, c.timestamp_ms) == (frame_id, ts)
        if flt is not None:
            assert flt.min_vertices <= c.vertex_count <= flt.max_vertices
            assert c.solidity >= flt.min_solidity - 1e-4                  # solidity is rounded to 4 places


def assert_lane_counts(rc, n_final):
    """Lane reject buckets sum to seen, and merging never adds candidates."""
    assert rc["seen"] == sum(rc[k] for k in LANE_BUCKETS), f"every contour must land in exactly one bucket: {rc}"
    assert rc["merged_into"] == n_final and n_final <= rc["accepted"]


def assert_sign_counts(rc, n_final):
    """Sign reject buckets sum to seen, and accepted matches what was returned."""
    assert rc["seen"] == sum(rc[k] for k in SIGN_BUCKETS), f"every contour must land in exactly one bucket: {rc}"
    assert rc["accepted"] == n_final


def assert_stop_line_candidates_ok(cands, shape, frame_id, ts, flt=TEST_STOP):
    """Invariants of every stop-line candidate: label, stamp, box inside the ROI, gates honored, nearest first."""
    h, w = shape
    for c in cands:
        assert c.label == STOP_LINE and (c.frame_id, c.timestamp_ms) == (frame_id, ts)
        x, y, bw, bh = c.bbox
        assert 0 <= x and 0 <= y and x + bw <= w + 1 and y + bh <= h and bw > 0 and bh > 0
        assert 0.0 <= c.confidence <= 1.0
        assert c.length_px >= flt.min_length_px and abs(c.tilt_deg) <= flt.max_tilt_deg
        assert flt.min_thickness_px <= c.thickness_px <= flt.max_thickness_px
        assert c.mean_intensity >= flt.min_intensity
        assert c.y_top_px <= c.y_bottom_px and c.y_near_px <= h
        assert c.clipped == (c.y_bottom_px == h)
    assert [c.y_near_px for c in cands] == sorted((c.y_near_px for c in cands), reverse=True)


def assert_stop_counts(rc, n_final):
    assert set(rc) == {"seen", *STOP_BUCKETS}
    assert rc["seen"] == sum(rc[b] for b in STOP_BUCKETS)
    assert rc["accepted"] == n_final


def assert_edge_map(edges, roi):
    """Edge map is a binary 0/255 image the size of its ROI."""
    assert edges.shape == roi.shape[:2] and edges.dtype == np.uint8
    assert set(np.unique(edges).tolist()) <= {0, 255}


def assert_geometry_contract(res, roi, lane_dbg, sign_dbg):
    """Everything the stage promises, checked against the ROICropResult it consumed."""
    assert isinstance(res, GeometryBranchResult)
    assert (res.frame_id, res.timestamp_ms) == (roi.frame_id, roi.timestamp_ms)
    assert_lane_candidates_ok(res.lane_candidates, roi.lane_roi.shape, res.frame_id, res.timestamp_ms)
    assert_sign_candidates_ok(res.sign_candidates, roi.sign_roi.shape, res.frame_id, res.timestamp_ms)
    assert_stop_line_candidates_ok(res.stop_line_candidates, roi.lane_roi.shape, res.frame_id, res.timestamp_ms)
    assert_lane_counts(lane_dbg["reject_counts"], len(res.lane_candidates))
    assert_sign_counts(sign_dbg["reject_counts"], len(res.sign_candidates))
    assert_stop_counts(lane_dbg["stop_line"]["reject_counts"], len(res.stop_line_candidates))
    assert_edge_map(lane_dbg["edges"], roi.lane_roi)
    assert_edge_map(sign_dbg["edges"], roi.sign_roi)


ROI_CONFIGS = {
    "default": ROIConfig(),
    "wide_lane": ROIConfig(lane=ROIBounds(0.0, 0.6, 1.0, 1.0)),
}


def build_scene(roi_cfg, frame_id=11, ts=222, H=FRAME_H, W=FRAME_W):
    """Full frame with a tape in the lane ROI and an octagon in the sign ROI, placed from the rects.

    Returns the cropped ROIs and the shapes' expected centers in frame coordinates.
    """
    frame = np.full((H, W, 3), BG, np.uint8)
    probe = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), roi_cfg)   # rects only
    lx, ly, lw, lh = probe.lane_rect
    sx, sy, sw, sh = probe.sign_rect
    tx, y0, y1 = lx + int(0.30 * lw), ly + int(0.15 * lh), ly + int(0.90 * lh)
    cv2.line(frame, (tx, y0), (tx + 4, y1), (FG,) * 3, 8)                    # near-vertical tape
    R, ox, oy = int(0.30 * min(sw, sh)), sx + sw // 2, sy + sh // 2
    pts = np.array([(ox + R*np.cos(np.pi/8 + 2*np.pi*k/8), oy + R*np.sin(np.pi/8 + 2*np.pi*k/8)) for k in range(8)], np.int32)
    cv2.fillPoly(frame, [pts], (FG,) * 3)                                    # octagon
    expect = {"tape_cx": tx + 2, "tape_cy": (y0 + y1) / 2, "sign_cx": ox, "sign_cy": oy}
    roi = crop_rois(preprocess_frame(FrameData(frame, frame_id, ts)), roi_cfg)
    return roi, expect


@pytest.mark.software
@pytest.mark.parametrize("bad", [None, np.empty((0, 1, 2), np.int32)])
def test_foot_x_of_nothing_is_the_sentinel(bad):
    assert contour_foot_x(bad) == -1.0


@pytest.mark.software
def test_foot_x_vertical_marking_is_the_base_center():
    assert contour_foot_x(cont([(10, 0), (10, 50), (14, 50), (14, 0)])) == 12.0


@pytest.mark.software
def test_foot_x_takes_only_the_lowest_rows():
    assert contour_foot_x(cont([(0, 0), (10, 10)])) == 10.0          # y >= 10-6 keeps only (10, 10)


@pytest.mark.software
@pytest.mark.parametrize("band", [0, -5])
def test_foot_x_band_of_zero_or_less_uses_only_the_lowest_row(band):
    assert contour_foot_x(cont([(0, 10), (4, 10), (2, 0)]), band_px=band) == 2.0


@pytest.mark.software
def test_foot_x_accepts_flat_and_opencv_shaped_contours_and_rounds():
    flat = np.array([[0, 5], [1, 5], [1, 5]], np.int32)               # (N, 2)
    assert contour_foot_x(flat) == contour_foot_x(flat.reshape(-1, 1, 2)) == round(2 / 3, 2)


@pytest.mark.software
def test_mean_intensity_of_a_filled_rectangle_is_its_value():
    img = blank((60, 100), 50)
    img[10:20, 30:60] = 200
    mask = np.zeros_like(img); mask[10:20, 30:60] = 255
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    assert geo._mean_contour_intensity(img, contour) == 200.0


@pytest.mark.software
def test_mean_intensity_of_a_contour_off_the_image_is_zero_and_partial_overlap_is_safe():
    img = blank((50, 100), 77)
    assert geo._mean_contour_intensity(img, cont([(500, 10), (510, 10), (510, 20), (500, 20)])) == 0.0
    part = geo._mean_contour_intensity(img, cont([(90, 10), (120, 10), (120, 20), (90, 20)]))
    assert part == 77.0


@pytest.mark.software
@pytest.mark.parametrize("kernel", [None, (), (1, 3), (0, 0)])
def test_close_edges_is_a_noop_for_trivial_kernels(kernel):
    edges = np.zeros((5, 5), np.uint8)
    assert geo._close_edges(edges, kernel) is edges


@pytest.mark.software
def test_close_edges_bridges_small_gaps_only():
    edges = np.zeros((9, 60), np.uint8)
    edges[4, 0:20] = 255
    edges[4, 25:40] = 255       # 5 px gap
    edges[4, 52:60] = 255       # 12 px gap before this run
    closed = geo._close_edges(edges, (9, 3))
    assert closed[4, 20:25].all()                 # 5 px gap bridged
    assert not closed[4, 41:52].all()             # 12 px gap survives
    assert np.all(closed[edges > 0] == 255)       # closing never removes an edge pixel


@pytest.mark.software
def test_extreme_points_are_leftmost_and_rightmost():
    left, right = geo._extreme_points(cont([(5, 9), (40, 2), (20, 30), (7, 1)]))
    assert left.tolist() == [5, 9] and right.tolist() == [40, 2]


def lane_conf(long=100, short=10, horiz=True, inten=200, roi_h=108, roi_w=432, f=TEST_LANE):
    """_lane_confidence with mid-range defaults, so each test varies one input."""
    return geo._lane_confidence(long, short, horiz, inten, roi_h, roi_w, f)


@pytest.mark.software
def test_lane_confidence_extremes():
    assert lane_conf(long=0, short=0, inten=TEST_LANE.min_intensity) == 0.0
    assert lane_conf(long=10_000, short=10_000, inten=255) == 1.0


@pytest.mark.software
@pytest.mark.parametrize("field", ["long", "short", "inten"])
def test_lane_confidence_never_decreases_with_better_measurements(field):
    ranges = {"long": range(0, 500, 10), "short": range(0, 60, 2), "inten": range(0, 256, 5)}
    scores = [lane_conf(**{field: v}) for v in ranges[field]]
    assert scores == sorted(scores) and scores[-1] > scores[0]


@pytest.mark.software
def test_lane_confidence_is_normalized_against_the_span_in_the_marking_direction():
    long = 0.25 * 100                          # a full ref_length of a 100 px span
    horizontal = lane_conf(long=long, short=0, inten=TEST_LANE.min_intensity, roi_w=100, roi_h=400, horiz=True)
    vertical = lane_conf(long=long, short=0, inten=TEST_LANE.min_intensity, roi_w=100, roi_h=400, horiz=False)
    assert horizontal > vertical


@pytest.mark.software
def test_lane_confidence_is_always_in_unit_range():
    rng = np.random.default_rng(3)
    for _ in range(500):
        s = lane_conf(*rng.uniform(0, 2000, 2), bool(rng.integers(0, 2)), rng.uniform(0, 300), *rng.uniform(1, 800, 2))
        assert 0.0 <= s <= 1.0


@pytest.mark.software
def test_sign_confidence_peaks_at_eight_vertices_and_is_symmetric_around_it():
    area = TEST_SIGN.ref_area
    by_v = {v: geo._sign_confidence(area, v, TEST_SIGN) for v in range(3, 14)}
    assert by_v[8] == max(by_v.values()) == 1.0
    assert by_v[7] == by_v[9] and by_v[6] == by_v[10]


@pytest.mark.software
def test_sign_confidence_grows_with_area_and_stays_in_unit_range():
    scores = [geo._sign_confidence(a, 8, TEST_SIGN) for a in range(0, 12000, 250)]
    assert scores == sorted(scores) and all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.software
@pytest.mark.parametrize("angle", [0.0, 3.0])
def test_horizontal_tape_is_found_where_it_was_drawn(angle):
    img = slanted_tape(cx=200, cy=80, length=200, thickness=8, angle_deg=angle)
    cands, dbg = extract_lane_candidates(img, TEST_CANNY, TEST_LANE, 5, 6, False)
    assert len(cands) == 1
    c = cands[0]
    x, y, w, h = c.bbox
    # Tolerances: Canny traces the outline just outside the fill, and closing grows it
    assert abs((x + w / 2) - 200) <= 6 and abs((y + h / 2) - 80) <= 6
    assert abs(c.length_px - 200) <= 10 and 6 <= c.width_px <= 14
    assert c.proximity == round(min((y + h) / img.shape[0], 1.0), 4)
    assert 150 <= c.mean_intensity <= 235            # the outline straddles the edge, so mat pixels pull it below FG
    assert 0.0 < c.confidence <= 1.0
    assert_lane_candidates_ok(cands, img.shape, 5, 6)
    assert_lane_counts(dbg["reject_counts"], 1)


@pytest.mark.software
def test_near_vertical_marking_foot_is_at_its_base_center():
    img = line_tape((210, 10), (210, 95))
    (c,), _ = extract_lane_candidates(img, TEST_CANNY, TEST_LANE, 1, 2, False)
    assert abs(c.foot_x - 210) <= 3


@pytest.mark.software
def test_diagonal_marking_foot_is_where_it_reaches_the_near_edge_not_the_bbox_middle():
    img = line_tape((50, 10), (150, 90))                       # runs down and to the right
    (c,), _ = extract_lane_candidates(img, TEST_CANNY, TEST_LANE, 1, 2, False)
    x, y, w, h = c.bbox
    assert abs(c.foot_x - 150) <= 10
    assert c.foot_x > (x + w / 2) + 25               # the bbox middle sits ~50 px left of the foot; 25 is half that


@pytest.mark.software
def test_dark_seam_on_a_bright_mat_is_rejected_by_the_intensity_gate():
    img = blank(LANE_SHAPE, 230)
    img[80:88, 100:300] = 20
    cands, dbg = extract_lane_candidates(img, TEST_CANNY, TEST_LANE, 1, 2, False)
    assert cands == [] and dbg["reject_counts"]["intensity"] == 1


@pytest.mark.software
@pytest.mark.parametrize("scene, override, gate", [
    ("h", dict(min_area=1e6), "area"),
    ("h", dict(max_aspect=2.0), "aspect"),
    ("h", dict(max_roi_span=0.3), "w_span"),
    ("v", dict(max_roi_span=0.3), "h_span"),
    ("h", dict(min_intensity=250.0), "intensity"),
])
def test_each_lane_gate_rejects_and_is_counted_under_its_own_name(scene, override, gate):
    img = slanted_tape() if scene == "h" else line_tape((210, 10), (210, 95))
    cands, dbg = extract_lane_candidates(img, TEST_CANNY, replace(TEST_LANE, **override), 1, 2, False)
    rc = dbg["reject_counts"]
    assert cands == [] and rc[gate] == 1 and rc["accepted"] == 0
    assert_lane_counts(rc, 0)


@pytest.mark.software
def test_contour_with_fewer_than_five_points_is_rejected_as_too_few():
    rc = {}
    out = geo._extract_lane_candidates([cont([(10, 10), (110, 10), (110, 20), (10, 20)])],
                                       TEST_LANE, 1, 2, LANE_SHAPE, blank(LANE_SHAPE, 200), rc)
    assert out == [] and rc["too_few_pts"] == 1


@pytest.mark.software
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (5, 5), (1, 50), (50, 1)])
def test_tiny_rois_do_not_raise(shape):
    roi = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    extract_lane_candidates(roi, TEST_CANNY, TEST_LANE, 1, 2, True)
    extract_sign_candidates(roi, TEST_CANNY, TEST_SIGN, 1, 2, True, True)


@pytest.mark.software
@pytest.mark.parametrize("value", [0, 255])
def test_featureless_rois_yield_nothing(value):
    roi = blank(LANE_SHAPE, value)
    lane, ld = extract_lane_candidates(roi, TEST_CANNY, TEST_LANE, 1, 2, True)
    sign, sd = extract_sign_candidates(roi, TEST_CANNY, TEST_SIGN, 1, 2, True, True)
    assert lane == [] and sign == [] and ld["reject_counts"]["seen"] == 0 and sd["trace"] == []


@pytest.mark.software
def test_lane_extraction_is_deterministic_and_leaves_the_roi_alone():
    img = slanted_tape()
    before = img.copy()
    a, da = extract_lane_candidates(img, TEST_CANNY, TEST_LANE, 1, 2, True)
    b, _ = extract_lane_candidates(img, TEST_CANNY, TEST_LANE, 1, 2, True)
    assert np.array_equal(img, before)
    assert [(c.bbox, c.confidence, c.foot_x) for c in a] == [(c.bbox, c.confidence, c.foot_x) for c in b]
    assert np.array_equal(da["lane_roi"], img)


@pytest.mark.software
def test_raw_edges_are_a_subset_of_the_closed_edges():
    _, dbg = extract_lane_candidates(noisy_scene(1, LANE_SHAPE), TEST_CANNY, TEST_LANE, 1, 2, False)
    assert_edge_map(dbg["edges"], dbg["lane_roi"]); assert_edge_map(dbg["edges_raw"], dbg["lane_roi"])
    assert np.all(dbg["edges"][dbg["edges_raw"] > 0] == 255)
    assert dbg["edges_lane"] is dbg["edges_raw"]                    # filter off: the same map


@pytest.mark.software
def test_filtered_lane_edges_are_a_subset_of_the_raw_edges_and_feed_the_closing():
    lane = stop_bar(100, 340, 40, 8, marks=(150, 290))
    _, dbg = extract_lane_candidates(lane, TEST_CANNY, TEST_LANE_FILTERED, 1, 2, False)
    assert np.all(dbg["edges_raw"][dbg["edges_lane"] > 0] == 255)
    assert (dbg["edges_lane"] > 0).sum() < (dbg["edges_raw"] > 0).sum()
    assert np.all(dbg["edges"][dbg["edges_lane"] > 0] == 255)


@pytest.mark.software
@pytest.mark.parametrize("seed", range(6))
def test_invariants_hold_on_cluttered_rois(seed):
    lane_img, sign_img = noisy_scene(seed, LANE_SHAPE), noisy_scene(seed + 100, SIGN_SHAPE)
    lane, ld = extract_lane_candidates(lane_img, TEST_CANNY, TEST_LANE, 9, 8, False)
    sign, sd = extract_sign_candidates(sign_img, TEST_CANNY, TEST_SIGN, 9, 8, False, True)
    assert_lane_candidates_ok(lane, LANE_SHAPE, 9, 8)
    assert_sign_candidates_ok(sign, SIGN_SHAPE, 9, 8, TEST_SIGN)
    assert_lane_counts(ld["reject_counts"], len(lane))
    assert_sign_counts(sd["reject_counts"], len(sign))


LANE_HW = LANE_SHAPE


@pytest.mark.software
def test_close_fragments_merge_into_one_candidate():
    (m,) = geo._merge_collinear([lane_cand(0, 50), lane_cand(60, 150, intensity=100.0)], TEST_LANE, LANE_HW)
    assert m.label == "lane_boundary" and m.bbox == (0, 20, 151, 4)
    assert m.length_px == pytest.approx(150.0, abs=1.0)
    assert m.contour.ndim == 3 and m.contour.shape[1:] == (1, 2)
    assert (m.frame_id, m.timestamp_ms) == (1, 2)
    assert 0.0 <= m.confidence <= 1.0 and 0.0 <= m.proximity <= 1.0 and m.foot_x >= 0


@pytest.mark.software
def test_merged_intensity_is_the_length_weighted_mean():
    (m,) = geo._merge_collinear([lane_cand(0, 50, intensity=200.0), lane_cand(60, 150, intensity=100.0)], TEST_LANE, LANE_HW)
    assert m.mean_intensity == pytest.approx((200 * 50 + 100 * 90) / 140, abs=1e-3)


@pytest.mark.software
@pytest.mark.parametrize("gap, merged", [(10, True), (39, True), (40, True), (41, False), (80, False)])
def test_merge_gap_threshold_is_inclusive_at_forty_pixels(gap, merged):
    out = geo._merge_collinear([lane_cand(0, 50), lane_cand(50 + gap, 120)], TEST_LANE, LANE_HW)
    assert len(out) == (1 if merged else 2)


@pytest.mark.software
def test_merge_gap_is_a_parameter():
    pair = [lane_cand(0, 50), lane_cand(65, 120)]                      # 15 px gap
    assert len(geo._merge_collinear(pair, TEST_LANE, LANE_HW, max_gap_px=10)) == 2
    assert len(geo._merge_collinear(pair, TEST_LANE, LANE_HW, max_gap_px=20)) == 1


@pytest.mark.software
def test_fragments_at_different_heights_are_not_merged():
    out = geo._merge_collinear([lane_cand(0, 50, y=5), lane_cand(60, 150, y=80)], TEST_LANE, LANE_HW)
    assert len(out) == 2


@pytest.mark.software
def test_a_chain_of_fragments_merges_into_one():
    (m,) = geo._merge_collinear([lane_cand(0, 30), lane_cand(35, 70), lane_cand(75, 110)], TEST_LANE, LANE_HW)
    assert m.bbox == (0, 20, 111, 4)


@pytest.mark.software
def test_vertical_candidates_pass_through_and_a_lone_candidate_is_untouched():
    vert = LaneCandidate("lane_boundary", (10, 10, 3, 80), cont([(10, 10), (13, 10), (13, 90), (10, 90)]), 0.5, 1, 2)
    lone = lane_cand(0, 50)
    out = geo._merge_collinear([lone, vert], TEST_LANE, LANE_HW)
    assert lone in out and vert in out and len(out) == 2
    mixed = geo._merge_collinear([lane_cand(0, 50), lane_cand(60, 150), vert], TEST_LANE, LANE_HW)
    assert len(mixed) == 2 and vert in mixed


@pytest.mark.software
def test_merging_zero_length_members_does_not_divide_by_zero():
    (m,) = geo._merge_collinear([lane_cand(0, 30, length=0.0), lane_cand(35, 70, length=0.0)], TEST_LANE, LANE_HW)
    assert m.mean_intensity == 0.0


@pytest.mark.software
def test_merging_never_increases_the_candidate_count():
    for seed in range(4):
        cands, dbg = extract_lane_candidates(noisy_scene(seed, LANE_SHAPE), TEST_CANNY, TEST_LANE, 1, 2, False)
        assert dbg["reject_counts"]["merged_into"] <= dbg["reject_counts"]["accepted"]


@pytest.mark.software
def test_octagon_is_found_where_it_was_drawn():
    img = regular_polygon(8, radius=40, center=(120, 100))
    (c,), dbg = extract_sign_candidates(img, TEST_CANNY, TEST_SIGN, 5, 6, False)
    x, y, w, h = c.bbox
    a = 2 * 40 * np.sin(np.pi / 8)
    assert abs((x + w / 2) - 120) <= 4 and abs((y + h / 2) - 100) <= 4
    # Regular octagon area 2(1 + sqrt 2)a^2 for side a; 10% for the traced outline sitting off the fill
    assert c.area == pytest.approx(2 * (1 + np.sqrt(2)) * a * a, rel=0.10)
    assert 8 <= c.vertex_count <= 10 and c.solidity >= 0.9 and c.confidence > 0.5
    assert_sign_candidates_ok([c], img.shape, 5, 6, TEST_SIGN)
    assert_sign_counts(dbg["reject_counts"], 1)


@pytest.mark.software
@pytest.mark.parametrize("sides", [3, 4, 6])
def test_polygons_with_the_wrong_vertex_count_are_rejected_as_vertices(sides):
    cands, dbg = extract_sign_candidates(regular_polygon(sides), TEST_CANNY, TEST_SIGN, 1, 2, False)
    assert cands == [] and dbg["reject_counts"]["vertices"] == 1


def run_sign_gates(contour, flt=TEST_SIGN):
    """One contour straight through the sign gates; returns (candidates, reject_counts, trace)."""
    rc, trace = {}, []
    out = geo._extract_sign_candidates([contour], flt, 3, 4, rc, trace)
    return out, rc, trace


@pytest.mark.software
def test_direct_octagon_contour_is_accepted_with_consistent_fields():
    oct_c = octagon_contour()
    (c,), rc, trace = run_sign_gates(oct_c)
    assert c.vertex_count == len(c.contour) == 8
    assert c.area == cv2.contourArea(oct_c) and c.solidity == pytest.approx(1.0, abs=1e-3)
    assert (c.frame_id, c.timestamp_ms) == (3, 4)
    assert [t["gate"] for t in trace] == [None]
    assert_sign_counts(rc, 1)


@pytest.mark.software
@pytest.mark.parametrize("name, contour, gate, traced", [
    ("square", SQUARE, "vertices", True),
    ("star", star_contour(), "solidity", True),
    ("tiny", TINY, "area", False),        # bbox smaller than min_area: counted, not traced
    ("sliver", SLIVER, "area", True),     # tiny area but sign-sized bbox: this is what the trace is for
])
def test_each_sign_gate_rejects_and_traces_as_documented(name, contour, gate, traced):
    out, rc, trace = run_sign_gates(contour)
    assert out == [] and rc[gate] == 1 and rc["accepted"] == 0
    assert_sign_counts(rc, 0)
    assert [t["gate"] for t in trace] == ([gate] if traced else [])
    for t in trace:
        assert set(t) == TRACE_KEYS


@pytest.mark.software
def test_trace_entries_are_none_for_fields_not_measured_before_rejection():
    _, _, trace = run_sign_gates(SLIVER)
    t = trace[0]
    assert t["vertices"] is None and t["solidity"] is None and t["confidence"] is None and t["poly"] is None
    assert t["area"] == cv2.contourArea(SLIVER)


@pytest.mark.software
def test_trace_is_only_present_when_requested():
    img = regular_polygon(8)
    _, off = extract_sign_candidates(img, TEST_CANNY, TEST_SIGN, 1, 2, False, trace=False)
    _, on = extract_sign_candidates(img, TEST_CANNY, TEST_SIGN, 1, 2, False, trace=True)
    assert "trace" not in off
    assert isinstance(on["trace"], list) and [t["gate"] for t in on["trace"]] == [None]


@pytest.mark.software
def test_overlays_exist_only_when_requested_and_are_three_channel():
    lane_img, sign_img = slanted_tape(), regular_polygon(8)
    for fn, roi, args in ((extract_lane_candidates, lane_img, (TEST_LANE,)), (extract_sign_candidates, sign_img, (TEST_SIGN,))):
        _, off = fn(roi, TEST_CANNY, *args, 1, 2, False)
        _, on = fn(roi, TEST_CANNY, *args, 1, 2, True)
        assert "contour_overlay" not in off and "accepted_overlay" not in off
        for key in ("contour_overlay", "accepted_overlay"):
            assert on[key].shape == roi.shape + (3,) and on[key].dtype == np.uint8


@pytest.mark.software
def test_accepted_overlays_carry_the_color_annotations():
    # A single-channel overlay keeps only the first BGR component and renders annotations black
    _, ld = extract_lane_candidates(slanted_tape(), TEST_CANNY, TEST_LANE, 1, 2, True)
    _, sd = extract_sign_candidates(regular_polygon(8), TEST_CANNY, TEST_SIGN, 1, 2, True)
    has = lambda img, bgr: bool(np.any(np.all(img == bgr, axis=2)))
    assert has(ld["accepted_overlay"], (0, 255, 0))
    assert has(sd["accepted_overlay"], (0, 0, 255))


def run_branch(lane, sign, **kw):
    """run_geometry_branch with the explicit test configs."""
    return run_geometry_branch(lane, sign, TEST_CANNY, TEST_LANE, TEST_SIGN, **kw)


@pytest.mark.software
@pytest.mark.parametrize("which", ["lane_roi", "sign_roi"])
@pytest.mark.parametrize("bad, exc", [
    (None, ValueError),
    (np.zeros((20, 20), np.float32), TypeError),
    (np.zeros((20, 20, 3), np.uint8), ValueError),
    (np.zeros(20, np.uint8), ValueError),
])
def test_invalid_rois_are_rejected_and_the_offender_is_named(which, bad, exc):
    good = blank((40, 40))
    lane, sign = (bad, good) if which == "lane_roi" else (good, bad)
    with pytest.raises(exc, match=which):
        run_branch(lane, sign)


@pytest.mark.software
def test_stage_carries_identity_and_returns_the_documented_triple():
    pre = PreprocessResult(gray=slanted_tape(shape=(FRAME_H, FRAME_W)), color=np.zeros((FRAME_H, FRAME_W, 3), np.uint8), frame_id=987654, timestamp_ms=555)
    roi = crop_rois(pre)
    out = run_geometry_stage(roi, TEST_CFG)
    assert len(out) == 3
    res, ld, sd = out
    assert (res.frame_id, res.timestamp_ms) == (987654, 555)
    assert isinstance(ld, dict) and isinstance(sd, dict)


@pytest.mark.software
def test_stage_is_the_branch_with_the_configs_unpacked():
    roi, _ = build_scene(ROIConfig())
    a, _, _ = run_geometry_stage(roi, TEST_CFG)
    b, _, _ = run_geometry_branch(roi.lane_roi, roi.sign_roi, TEST_CANNY, TEST_LANE, TEST_SIGN, roi.frame_id, roi.timestamp_ms, False)
    key = lambda r: ([(c.bbox, c.confidence, c.foot_x) for c in r.lane_candidates], [(c.bbox, c.confidence) for c in r.sign_candidates])
    assert key(a) == key(b)


@pytest.mark.software
def test_each_config_component_reaches_its_own_branch():
    roi, _ = build_scene(ROIConfig())
    base, _, _ = run_geometry_stage(roi, TEST_CFG)
    assert len(base.lane_candidates) >= 1 and len(base.sign_candidates) >= 1

    no_lane, _, _ = run_geometry_stage(roi, replace(TEST_CFG, lane=replace(TEST_LANE, min_area=1e9)))
    assert no_lane.lane_candidates == [] and len(no_lane.sign_candidates) == len(base.sign_candidates)

    no_sign, _, _ = run_geometry_stage(roi, replace(TEST_CFG, sign=replace(TEST_SIGN, min_vertices=20, max_vertices=30)))
    assert no_sign.sign_candidates == [] and len(no_sign.lane_candidates) == len(base.lane_candidates)


@pytest.mark.software
def test_overlays_and_trace_are_off_by_default_at_the_stage_level():
    roi, _ = build_scene(ROIConfig())
    _, ld, sd = run_geometry_stage(roi, TEST_CFG)
    assert "accepted_overlay" not in ld and "accepted_overlay" not in sd and "trace" not in sd
    _, ld, sd = run_geometry_stage(roi, TEST_CFG, draw_overlays=True, trace=True)
    assert "accepted_overlay" in ld and "accepted_overlay" in sd and "trace" in sd


@pytest.mark.software
def test_default_configs_do_not_share_mutable_inner_objects():
    a, b = GeometryConfig(), GeometryConfig()
    assert a.canny is not b.canny and a.lane is not b.lane and a.sign is not b.sign


@pytest.mark.software
@pytest.mark.parametrize("roi_cfg", ROI_CONFIGS.values(), ids=ROI_CONFIGS.keys())
def test_chain_finds_shapes_at_their_frame_coordinates(roi_cfg):
    roi, exp = build_scene(roi_cfg)
    assert not roi.lane_roi.flags.writeable                     # geometry must accept the read-only views
    res, ld, sd = run_geometry_stage(roi, TEST_CFG)
    assert_geometry_contract(res, roi, ld, sd)

    lx, ly = roi.lane_rect[:2]
    sx, sy = roi.sign_rect[:2]
    assert len(res.lane_candidates) == 1 and len(res.sign_candidates) == 1
    x, y, w, h = res.lane_candidates[0].bbox                    # ROI-relative -> add the rect origin
    # Wider than the direct tests: preprocess blurs the shapes before Canny sees them
    assert abs(lx + x + w / 2 - exp["tape_cx"]) <= 8 and abs(ly + y + h / 2 - exp["tape_cy"]) <= 8
    x, y, w, h = res.sign_candidates[0].bbox
    assert abs(sx + x + w / 2 - exp["sign_cx"]) <= 6 and abs(sy + y + h / 2 - exp["sign_cy"]) <= 6


@pytest.mark.software
def test_shipped_default_config_runs_and_honors_the_contract():
    roi, _ = build_scene(ROIConfig())
    res, ld, sd = run_geometry_stage(roi)                       # defaults; counts are tuning, contract is not
    assert_geometry_contract(res, roi, ld, sd)


@pytest.mark.software
def test_every_recorded_frame_meets_the_geometry_contract(dataset_frames):
    if not dataset_frames:
        pytest.skip("no recorded dataset in tests/data/frames (run: pytest --hardware --record)")
    for fd in dataset_frames:
        roi = crop_rois(preprocess_frame(fd))
        try:
            assert_geometry_contract(*(lambda r: (r[0], roi, r[1], r[2]))(run_geometry_stage(roi)))
        except AssertionError as e:
            raise AssertionError(f"frame_id={fd.frame_id}: {e}") from e


# =============================================================================
# Stop lines
# =============================================================================

def stop_bar(x0=100, x1=300, y=40, thickness=8, tilt_deg=0.0, value=FG, marks=(), mark_width=8,
             shape=LANE_SHAPE):
    """A bar across a lane ROI, optionally over vertical lane marks; tilt rotates it about its center (+ = right end lower)."""
    img = blank(shape)
    for mx in marks:
        cv2.rectangle(img, (mx - mark_width // 2, 0), (mx + mark_width // 2, shape[0] - 1), FG, -1)
    if tilt_deg == 0.0:
        cv2.rectangle(img, (x0, y), (x1, y + thickness - 1), value, -1)
    else:
        box = cv2.boxPoints((((x0 + x1) / 2.0, y + thickness / 2.0), (x1 - x0, thickness), tilt_deg))
        cv2.fillPoly(img, [np.round(box).astype(np.int32)], value)
    return img


def stop_lines(lane, flt=TEST_STOP, frame_id=4, ts=200):
    """The stop-line detector on a lane ROI, reading the Canny map the lane detector would."""
    return extract_stop_line_candidates(lane, geo._canny(lane, TEST_CANNY), flt, frame_id, ts)


@pytest.mark.software
def test_a_stop_line_is_found_where_it_was_drawn():
    cands, dbg = stop_lines(stop_bar(100, 300, 40, 8))
    assert len(cands) == 1
    c = cands[0]
    assert abs(c.x_left - 100) <= 3 and abs(c.x_right - 300) <= 3
    assert abs(c.y_top_px - 40) <= 1.5 and abs(c.y_bottom_px - 48) <= 1.5
    assert abs(c.thickness_px - 8) <= 1.5 and abs(c.tilt_deg) < 0.5
    assert c.y_near_px == pytest.approx(c.y_bottom_px, abs=0.5) and not c.clipped
    assert (c.frame_id, c.timestamp_ms) == (4, 200)
    assert_stop_line_candidates_ok(cands, LANE_SHAPE, 4, 200)
    assert_stop_counts(dbg["reject_counts"], 1)


@pytest.mark.software
def test_a_stop_line_touching_lane_lines_is_found_as_one_line_across_them():
    """The gradient split keeps the lane lines' sides out, so the bar comes out alone."""
    cands, _ = stop_lines(stop_bar(100, 340, 40, 8, marks=(150, 290)))
    assert len(cands) == 1
    assert abs(cands[0].x_left - 100) <= 3 and abs(cands[0].x_right - 340) <= 3


@pytest.mark.software
def test_a_stop_line_broken_by_wide_lane_tape_is_joined_into_one():
    """30 px tape breaks the bar's edges wider than the closing bridges; the pieces are refit as one."""
    cands, _ = stop_lines(stop_bar(80, 380, 40, 10, marks=(150, 300), mark_width=30))
    assert len(cands) == 1
    assert cands[0].x_left < 120 and cands[0].x_right > 340


@pytest.mark.software
@pytest.mark.parametrize("tilt", [8.0, -8.0])
def test_tilt_is_measured_with_its_sign_and_the_nearer_end_sets_y_near(tilt):
    c, = stop_lines(stop_bar(120, 320, 45, 8, tilt_deg=tilt))[0]
    assert c.tilt_deg == pytest.approx(tilt, abs=1.5)
    assert c.y_near_px > c.y_bottom_px + 5          # the lower end is well below the midpoint


@pytest.mark.software
def test_a_line_steeper_than_max_tilt_never_reaches_the_detector():
    cands, dbg = stop_lines(stop_bar(120, 320, 45, 8, tilt_deg=35.0))
    assert cands == [] and dbg["reject_counts"]["seen"] == 0


@pytest.mark.software
def test_a_line_running_off_the_roi_bottom_is_clipped_at_the_bottom_row():
    h = LANE_SHAPE[0]
    c, = stop_lines(stop_bar(100, 300, h - 10, 20))[0]
    assert c.clipped and c.y_near_px == h and c.y_bottom_px == h
    assert c.thickness_px == pytest.approx(10, abs=1.5)


@pytest.mark.software
@pytest.mark.parametrize("scene, gate", [
    (dict(x0=200, x1=240), "short"),                          # 40 px: a lane-tape width, not a stop line
    (dict(value=110), "intensity"),                           # edges found, but too dim for tape
    (dict(y=10, thickness=60), "unpaired"),                   # edges 60 px apart: thicker than any stop line
], ids=["short", "intensity", "unpaired"])
def test_each_stop_line_gate_rejects_and_is_counted_under_its_own_name(scene, gate):
    cands, dbg = stop_lines(stop_bar(**scene))
    assert cands == []
    assert dbg["reject_counts"][gate] >= 1
    assert_stop_counts(dbg["reject_counts"], 0)


@pytest.mark.software
def test_the_tilt_gate_rejects_an_edge_fitted_steeper_than_max_tilt():
    """The gradient split already drops most steep edges; the fitted line is gated again."""
    x = np.arange(0, 200, dtype=np.float32)
    steep = geo._fit_segment(np.stack([x, 20 + x * np.tan(np.radians(25))], axis=1))
    assert geo._stop_line_from(steep, [], blank(LANE_SHAPE), TEST_STOP, 0, 0) == (None, "tilt")


@pytest.mark.software
def test_the_stop_line_trace_has_one_entry_per_top_edge_agreeing_with_the_counts():
    lane = stop_bar(100, 300, 50, 8)
    cv2.rectangle(lane, (200, 10), (240, 17), FG, -1)                     # 40 px: short
    cands, dbg = stop_lines(lane)
    trace, rc = dbg["trace"], dbg["reject_counts"]
    assert len(trace) == rc["seen"] == 2
    for gate in STOP_BUCKETS[:-1]:
        assert sum(e["gate"] == gate for e in trace) == rc[gate]
    accepted = [e["candidate"] for e in trace if e["gate"] is None]
    assert accepted == cands and all(e["candidate"] is None for e in trace if e["gate"])
    assert len(dbg["bottoms"]) == dbg["bottom_count"]


@pytest.mark.software
def test_stop_lines_come_out_nearest_the_robot_first():
    lane = stop_bar(100, 300, 15, 8)
    cv2.rectangle(lane, (100, 70), (300, 77), FG, -1)
    cands, _ = stop_lines(lane)
    assert len(cands) == 2 and cands[0].y_near_px > cands[1].y_near_px


@pytest.mark.software
@pytest.mark.parametrize("seed", range(6))
def test_lane_marks_dashes_and_clutter_give_no_stop_line(seed):
    """Vertical tape, dash ends and noise: nothing lying across the ROI long enough to be a stop line."""
    lane = stop_bar(0, 1, 0, 1, value=BG, marks=(150, 290))
    cv2.rectangle(lane, (60, 10), (80, 40), FG, -1)                            # a dash: 20 px ends
    rng = np.random.default_rng(seed)
    lane = cv2.add(lane, rng.integers(0, 40, lane.shape, dtype=np.uint8))
    assert stop_lines(lane)[0] == []


@pytest.mark.software
def test_lane_and_sign_results_do_not_depend_on_the_stop_line_filter():
    """Stop lines only read the shared Canny map: any StopLineFilter leaves lanes and signs as they are."""
    from src.tests.scenes import SCENE_CONFIG, SCENES
    permissive = StopLineFilter(max_tilt_deg=45.0, min_length_px=5.0, min_thickness_px=0.5,
                                max_thickness_px=200.0, min_intensity=0.0, close_kernel=(31, 3))
    closed = StopLineFilter(min_length_px=1e6)
    for name, frame in SCENES.items():
        roi = crop_rois(preprocess_frame(FrameData(frame, 1, 50), SCENE_CONFIG.preprocess), SCENE_CONFIG.roi)
        runs = [run_geometry_stage(roi, replace(SCENE_CONFIG.geometry, stop_line=f)) for f in (permissive, closed)]
        (a, a_lane, a_sign), (b, b_lane, b_sign) = runs
        assert len(b.stop_line_candidates) == 0, name
        for x, y in ((a.lane_candidates, b.lane_candidates), (a.sign_candidates, b.sign_candidates)):
            assert len(x) == len(y), name
            for cx, cy in zip(x, y):
                assert cx.bbox == cy.bbox and cx.confidence == cy.confidence, name
        assert a_lane["reject_counts"] == b_lane["reject_counts"], name
        assert a_sign["reject_counts"] == b_sign["reject_counts"], name


@pytest.mark.software
def test_shared_canny_edges_give_the_lane_detector_what_it_computes_itself():
    lane = stop_bar(100, 340, 40, 8, marks=(150, 290))
    edges = geo._canny(lane, TEST_CANNY)
    own, own_dbg = extract_lane_candidates(lane, TEST_CANNY, TEST_LANE, 1, 2, False)
    shared, shared_dbg = extract_lane_candidates(lane, TEST_CANNY, TEST_LANE, 1, 2, False, edges_raw=edges)
    assert [(c.bbox, c.confidence) for c in own] == [(c.bbox, c.confidence) for c in shared]
    assert own_dbg["reject_counts"] == shared_dbg["reject_counts"]
    assert [c.bbox for c in find_lane_candidates(lane, TEST_CANNY, TEST_LANE, 1, 2, edges_raw=edges)] \
        == [c.bbox for c in own]


@pytest.mark.software
def test_stop_line_config_reaches_the_detector_through_the_stage():
    lane = stop_bar(100, 300, 40, 8)
    roi = SimpleNamespace(lane_roi=lane, sign_roi=blank(SIGN_SHAPE), frame_id=3, timestamp_ms=30)
    found, _, _ = run_geometry_stage(roi, TEST_CFG)
    refused, _, _ = run_geometry_stage(roi, replace(TEST_CFG, stop_line=replace(TEST_STOP, min_length_px=500.0)))
    assert len(found.stop_line_candidates) == 1 and refused.stop_line_candidates == []
    assert geo.detect_geometry(roi, TEST_CFG).stop_line_candidates[0].bbox == found.stop_line_candidates[0].bbox


# =============================================================================
# Lines across the lane, out of the lane detector's edges
# =============================================================================

def lanes_found(lane, flt=TEST_LANE_FILTERED):
    return extract_lane_candidates(lane, TEST_CANNY, flt, 1, 2, False)


# Unblurred drawn ROIs, so thin marks: wide tape needs preprocess's blur to
# trace as a shape, and is covered through the pipeline in test_pipeline
@pytest.mark.software
def test_a_stop_line_touching_the_lane_lines_no_longer_joins_them():
    lane = stop_bar(140, 300, 40, 10, marks=(150, 290), mark_width=6)
    joined, _ = lanes_found(lane, TEST_LANE)
    assert len(joined) == 1 and joined[0].bbox[2] > 150                # one H-shaped contour
    apart, _ = lanes_found(lane)
    assert apart and all(c.bbox[2] < 45 for c in apart)                 # the lane lines, apart
    assert any(c.bbox[0] <= 150 <= c.bbox[0] + c.bbox[2] for c in apart)
    assert any(c.bbox[0] <= 290 <= c.bbox[0] + c.bbox[2] for c in apart)


@pytest.mark.software
def test_the_stubs_of_a_stop_line_running_past_the_tape_go_too():
    """Shorter than a tape end on their own; removed because they lie on the long run's line."""
    lane = stop_bar(40, 400, 40, 10, marks=(150, 290), mark_width=6)
    cands, _ = lanes_found(lane)
    assert cands and all(c.bbox[2] < 45 for c in cands)


@pytest.mark.software
def test_the_ends_of_a_piece_of_tape_stay_so_it_traces_as_one():
    dash = blank(LANE_SHAPE)
    cv2.rectangle(dash, (140, 20), (160, 60), FG, -1)                   # 20 px wide, 40 px long
    plain, _ = lanes_found(dash, TEST_LANE)
    filtered, dbg = lanes_found(dash)
    assert dbg["edges_lane"] is dbg["edges_raw"]                        # no run long enough to remove
    assert [(c.bbox, c.confidence) for c in filtered] == [(c.bbox, c.confidence) for c in plain]


@pytest.mark.software
def test_a_tilted_stop_line_within_the_angle_goes_and_a_steeper_line_stays():
    # Blurred as preprocess does: a drawn tilted edge is a pixel staircase whose
    # steps read steeper than the angle and break the run into short pieces
    tilted_bar = cv2.GaussianBlur(stop_bar(100, 340, 45, 10, tilt_deg=12.0, marks=(150, 290)), (9, 3), 0)
    tilted, _ = lanes_found(tilted_bar)
    assert tilted and all(c.bbox[2] < 60 for c in tilted)
    _, dbg = lanes_found(line_tape((120, 100), (300, 10), thickness=8))   # ~27 deg: a line the lane keeps
    assert dbg["edges_lane"] is dbg["edges_raw"]


@pytest.mark.software
def test_lane_lines_alone_are_untouched_by_the_filter():
    _, dbg = lanes_found(stop_bar(0, 1, 0, 1, value=BG, marks=(150, 290), mark_width=20))
    assert dbg["edges_lane"] is dbg["edges_raw"]


@pytest.mark.software
def test_the_stop_line_detector_reads_the_full_edges_whatever_the_lane_filter():
    from src.tests.scenes import SCENE_CONFIG, SCENES
    roi = crop_rois(preprocess_frame(FrameData(SCENES["stop_line_touching"], 1, 50), SCENE_CONFIG.preprocess),
                    SCENE_CONFIG.roi)
    on = run_geometry_stage(roi, SCENE_CONFIG.geometry)[0]
    off = run_geometry_stage(roi, replace(SCENE_CONFIG.geometry,
                                          lane=replace(SCENE_CONFIG.geometry.lane, horizontal_edge_deg=None)))[0]
    assert [c.bbox for c in on.stop_line_candidates] == [c.bbox for c in off.stop_line_candidates]
    assert len(on.stop_line_candidates) == 1
    narrow = lambda cands: [c for c in cands if c.bbox[2] < 45]
    assert narrow(off.lane_candidates) == [] and len(narrow(on.lane_candidates)) >= 2   # joined vs apart


@pytest.mark.software
def test_the_lane_filter_is_the_same_whether_it_shares_the_stop_line_split_or_not():
    """At the same angle the stop-line detector's split is reused; at another the lane computes its own."""
    lane = stop_bar(100, 340, 40, 10, marks=(150, 290), mark_width=20)
    edges = geo._canny(lane, TEST_CANNY)
    shared = geo._strip_horizontal_lines(lane, edges, TEST_LANE_FILTERED, geo._horizontal_edges(lane, edges, 20.0))
    own = geo._strip_horizontal_lines(lane, edges, TEST_LANE_FILTERED)
    assert np.array_equal(shared, own)
    split = geo._splits(lane, edges, TEST_LANE_FILTERED, replace(TEST_STOP, max_tilt_deg=15.0))
    assert split[1] is None


def _jsonable_trace(trace):
    """Sign trace with numpy bboxes and polygons converted to lists for JSON."""
    out = []
    for t in trace:
        t = dict(t)
        t["bbox"] = list(t["bbox"])
        t["poly"] = None if t["poly"] is None else t["poly"].reshape(-1, 2).tolist()
        out.append(t)
    return out


@pytest.mark.hardware
def test_geometry_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    cfg = GeometryConfig()                                       # the shipped tuning

    rows, samples = [], {}
    gate_totals = {"lane": {}, "sign": {}}
    for i, fd in enumerate(frames(n)):
        roi = crop_rois(preprocess_frame(fd))                    # upstream, not timed

        t0 = time.perf_counter_ns()
        res, ld, sd = run_geometry_stage(roi, cfg)               # live-loop settings: no overlays, no trace
        stage_ms = (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        extract_lane_candidates(roi.lane_roi, cfg.canny, cfg.lane, fd.frame_id, fd.timestamp_ms, False)
        lane_ms = (time.perf_counter_ns() - t0) / 1e6
        t0 = time.perf_counter_ns()
        extract_sign_candidates(roi.sign_roi, cfg.canny, cfg.sign, fd.frame_id, fd.timestamp_ms, False)
        sign_ms = (time.perf_counter_ns() - t0) / 1e6

        assert_geometry_contract(res, roi, ld, sd)               # outside the timing window
        lrc, src = ld["reject_counts"], sd["reject_counts"]
        for k, v in lrc.items():
            gate_totals["lane"][k] = gate_totals["lane"].get(k, 0) + v
        for k, v in src.items():
            gate_totals["sign"][k] = gate_totals["sign"].get(k, 0) + v
        rows.append((fd.frame_id, fd.timestamp_ms, round(stage_ms, 3), round(lane_ms, 3), round(sign_ms, 3),
                     len(res.lane_candidates), len(res.sign_candidates),
                     *(lrc[k] for k in ("seen",) + LANE_BUCKETS), *(src[k] for k in ("seen",) + SIGN_BUCKETS)))
        if i in (0, n // 2, n - 1):
            samples[fd.frame_id] = roi

    if not rows:
        pytest.skip("no frames delivered")

    header = (["frame_id", "timestamp_ms", "stage_ms", "lane_ms", "sign_ms", "n_lane", "n_sign"]
              + [f"lane_{k}" for k in ("seen",) + LANE_BUCKETS] + [f"sign_{k}" for k in ("seen",) + SIGN_BUCKETS])
    stage = [r[2] for r in rows]
    artifacts.json("config.json", asdict(cfg))
    artifacts.csv("geometry_timing.csv", header, rows)
    artifacts.json("summary.json", {
        "stage_ms": summarize(stage), "lane_ms": summarize(r[3] for r in rows),
        "sign_ms": summarize(r[4] for r in rows), "frames_over_33ms": sum(1 for s in stage if s > 33.3),   # one frame period at MAX_FPS (30)
        "frames_with_lane": sum(1 for r in rows if r[5]), "frames_with_sign": sum(1 for r in rows if r[6]),
        "gate_totals": gate_totals,
    })
    artifacts.histogram("stage_ms_hist.png", stage, "run_geometry_stage latency", "ms")

    for fid, roi in samples.items():                             # untimed, with the debug views
        res, ld, sd = run_geometry_stage(roi, cfg, draw_overlays=True, trace=True)
        for name, img in (("lane_edges_raw", ld["edges_raw"]), ("lane_edges", ld["edges"]),
                          ("lane_contours", ld["contour_overlay"]), ("lane_accepted", ld["accepted_overlay"]),
                          ("sign_edges", sd["edges"]), ("sign_contours", sd["contour_overlay"]),
                          ("sign_accepted", sd["accepted_overlay"])):
            artifacts.image(f"{fid:06d}_{name}.png", img)
        artifacts.json(f"{fid:06d}_sign_trace.json", _jsonable_trace(sd["trace"]))
        artifacts.json(f"{fid:06d}_candidates.json", {
            "lane": [{"bbox": list(c.bbox), "confidence": c.confidence, "proximity": c.proximity, "foot_x": c.foot_x,
                      "length_px": c.length_px, "width_px": c.width_px, "mean_intensity": round(c.mean_intensity, 2)}
                     for c in res.lane_candidates],
            "sign": [{"bbox": list(c.bbox), "confidence": c.confidence, "vertices": c.vertex_count,
                      "area": c.area, "solidity": c.solidity} for c in res.sign_candidates],
        })