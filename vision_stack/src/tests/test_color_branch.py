"""
test_color_branch.py  --  src/perception/color_branch.py

Detection tests use their own explicit HSV bands and blob filter (never the
shipped scaffold or your calibration), synthetic ROIs with known blob positions,
and hand-drawn masks for the blob gates. The chained tests place blobs from
crop_rois()'s traffic_rect, so they hold for any ROI bounds. Glow mode gets
LEDs drawn as a clipped white center in a colored ring, beside unlit lenses.

--software  Contract, gate-wiring, loader, known-answer and chained tests.
--hardware  Times run_color_stage per frame (live or --replay), using your
            calibrated ranges if calibration/hsv_ranges.json exists (scaffold
            otherwise, and the run says so), and writes CSV, masks and overlays.
"""
import json
import time
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.capture.camera import FrameData
from src.perception import color_branch as cb
from src.perception.color_branch import (
    BlobFilter, ColorConfig, ColorRange, GlowFilter, HSVRanges, TrafficLightCandidate,
    detect_color, draw_candidates, extract_traffic_light_candidates, find_traffic_light_candidates,
    load_color_config, load_hsv_ranges, run_color_stage,
)
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import ROIBounds, ROIConfig, crop_rois
from src.params import FRAME_H, FRAME_W
from src.tests.artifacts import summarize

# Explicit rather than the scaffold or a calibration, so retuning never breaks detection tests
TEST_HSV = HSVRanges(
    red_low=ColorRange((0, 120, 120), (10, 255, 255)),
    red_high=ColorRange((170, 120, 120), (180, 255, 255)),
    yellow=ColorRange((20, 120, 120), (35, 255, 255)),
    green=ColorRange((40, 120, 120), (80, 255, 255)))
# The roundness and core gates off: these tests draw solid shapes to test the other gates;
# the lamp-vs-shirt tests below run the production gates on lamps drawn with a clipped core
TEST_BLOB = BlobFilter(min_area=50.0, max_area=5000.0, min_aspect=0.3, max_aspect=3.0, ref_area=800.0, min_roundness=0.0, min_core_px=0)
TEST_CFG = ColorConfig(TEST_HSV, TEST_BLOB)

LABELS = ("red", "yellow", "green")
PURE = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}   # BGR
BG = 20                                                                       # dark background, below every band's V floor
GATES = ("seen", "area", "aspect", "round", "core", "accepted")
TRACE_KEYS = {"label", "bbox", "gate", "area", "aspect", "fill", "confidence", "roundness", "core_px", "hsv"}
CALIB = Path(__file__).resolve().parents[2] / "calibration" / "hsv_ranges.json"


def bgr_from_hsv(h, s=255, v=255):
    """BGR tuple for one OpenCV HSV color."""
    return tuple(int(c) for c in cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0])


def scene(shape=(120, 240), blobs=()):
    """Dark ROI with filled circles: blobs = [(color, cx, cy, radius)], color a label or a BGR tuple."""
    img = np.full(shape + (3,), BG, np.uint8)
    for color, cx, cy, r in blobs:
        cv2.circle(img, (cx, cy), r, PURE.get(color, color) if isinstance(color, str) else color, -1)
    return img


THREE = [("red", 40, 60, 10), ("yellow", 120, 60, 10), ("green", 200, 60, 10)]


def rect_mask(shape, *rects):
    """0/255 mask with filled (x, y, w, h) rectangles."""
    m = np.zeros(shape, np.uint8)
    for x, y, w, h in rects:
        m[y:y + h, x:x + w] = 255
    return m


def blobs_of(mask, flt=TEST_BLOB, label="red", trace=None, hsv=None, rc=None):
    """One mask straight through the blob gates; returns (candidates, reject_counts)."""
    rc = {} if rc is None else rc
    return cb._blobs_to_candidates(mask, label, flt, 3, 4, rc, trace, hsv), rc


def hsv_json(hsv=TEST_HSV):
    """HSVRanges in the calibration file's JSON layout."""
    return {k: {"lower": list(getattr(hsv, k).lower), "upper": list(getattr(hsv, k).upper)}
            for k in ("red_low", "red_high", "yellow", "green")}


def assert_color_contract(cands, dbg, fid, ts):
    """Everything the extractor documents about its candidates and debug dict."""
    h, w = dbg["roi"].shape[:2]
    for c in cands:
        x, y, bw, bh = c.bbox
        assert isinstance(c, TrafficLightCandidate) and c.label in LABELS
        assert x >= 0 and y >= 0 and bw >= 1 and bh >= 1 and x + bw <= w and y + bh <= h, f"bbox {c.bbox} outside {(h, w)}"
        assert 0.0 <= c.confidence <= 1.0
        assert (c.frame_id, c.timestamp_ms) == (fid, ts)
    assert dbg["hsv"].shape == (h, w, 3) and dbg["hsv"].dtype == np.uint8
    for color in LABELS:
        m = dbg[color]
        assert m.shape == (h, w) and m.dtype == np.uint8 and set(np.unique(m).tolist()) <= {0, 255}
        assert dbg["mask_px"][color] == int(np.count_nonzero(m))
        rc = dbg["reject_counts"][color]
        assert set(rc) == set(GATES)
        assert rc["seen"] == rc["area"] + rc["aspect"] + rc["round"] + rc["core"] + rc["accepted"], f"{color}: {rc}"
        assert rc["accepted"] == sum(1 for c in cands if c.label == color)
    assert isinstance(dbg["calibrated"], bool)


ROI_CONFIGS = {
    "default": ROIConfig(),
    "narrow_traffic": ROIConfig(traffic=ROIBounds(0.30, 0.05, 0.70, 0.45)),
}


# Lamp radius in px: ~200 px^2, over the test blob gates' min_area, and fits
# the default traffic ROI, 62 px tall since it was cut to the course's light (2026-10-06)
LAMP_R = 8


def build_traffic_scene(roi_cfg, frame_id=11, ts=222, H=FRAME_H, W=FRAME_W):
    """Colored blobs drawn inside the traffic rect; returns the crop result and the drawn blob centers."""
    frame = np.full((H, W, 3), BG, np.uint8)
    probe = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), roi_cfg)
    tx, ty, tw, th = probe.traffic_rect
    r = LAMP_R
    spots = {"red": (0.2, 0.3), "green": (0.8, 0.3), "yellow": (0.5, 0.65)}
    expect = {}
    for label, (fx, fy) in spots.items():
        cx, cy = tx + int(fx * tw), ty + int(fy * th)
        cv2.circle(frame, (cx, cy), r, PURE[label], -1)
        expect[label] = (cx, cy)
    roi = crop_rois(preprocess_frame(FrameData(frame, frame_id, ts)), roi_cfg)
    return roi, expect


@pytest.mark.software
def test_hsv_ranges_start_uncalibrated_and_instances_are_independent():
    a, b = HSVRanges(), HSVRanges()
    assert a.is_calibrated is False
    assert a.red_low is not b.red_low and a.green is not b.green


@pytest.mark.software
def test_color_config_default_leaves_the_branch_off_and_is_frozen():
    cfg = ColorConfig()
    assert cfg.hsv_ranges is None
    assert cfg.blob is not ColorConfig().blob
    with pytest.raises(FrozenInstanceError):
        cfg.hsv_ranges = TEST_HSV


def write_json(tmp_path, data):
    """Write a calibration dict to a temp file; returns its path."""
    p = tmp_path / "hsv.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.mark.software
def test_load_round_trips_every_band_and_marks_calibrated(tmp_path):
    r = load_hsv_ranges(write_json(tmp_path, hsv_json()))
    assert r.is_calibrated is True
    for band in ("red_low", "red_high", "yellow", "green"):
        got, want = getattr(r, band), getattr(TEST_HSV, band)
        assert (got.lower, got.upper) == (want.lower, want.upper)
        assert all(isinstance(v, int) for v in got.lower + got.upper)


@pytest.mark.software
def test_hue_upper_bound_of_180_is_accepted_for_the_red_wrap(tmp_path):
    data = hsv_json(); data["yellow"]["upper"] = [180, 255, 255]
    assert load_hsv_ranges(write_json(tmp_path, data)).yellow.upper[0] == 180


@pytest.mark.software
def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hsv_ranges(str(tmp_path / "nope.json"))


def _set(band, key, value):
    """Mutator that overwrites one bound in a calibration dict."""
    def mutate(d):
        d[band][key] = value
    return mutate


def _drop(band, key=None):
    """Mutator that removes a band, or one bound of it."""
    def mutate(d):
        d.pop(band) if key is None else d[band].pop(key)
    return mutate


@pytest.mark.software
@pytest.mark.parametrize("mutate, exc, match", [
    (_set("green", "lower", [40, 120]), ValueError, "green: lower and upper"),
    (_set("green", "lower", [90, 120, 120]), ValueError, "green: H"),        # lower hue above upper
    (_set("yellow", "upper", [181, 255, 255]), ValueError, "yellow: H"),
    (_set("yellow", "lower", [20, -1, 120]), ValueError, "yellow: S"),
    (_set("yellow", "upper", [35, 256, 255]), ValueError, "yellow: S"),
    (_set("red_low", "upper", [10, 255, 256]), ValueError, "red_low: V"),
    (_drop("red_high"), KeyError, "red_high"),
    (_drop("green", "upper"), KeyError, "upper"),
], ids=["2_values", "lower_gt_upper", "hue_181", "s_negative", "s_256", "v_256", "missing_band", "missing_bound"])
def test_malformed_calibration_fails_at_load_and_names_the_problem(tmp_path, mutate, exc, match):
    data = hsv_json(); mutate(data)
    with pytest.raises(exc, match=match):
        load_hsv_ranges(write_json(tmp_path, data))


@pytest.mark.software
def test_load_color_config_wires_ranges_and_blob(tmp_path):
    path = write_json(tmp_path, hsv_json())
    default = load_color_config(path)
    assert default.hsv_ranges.is_calibrated and default.blob == BlobFilter()
    custom = BlobFilter(min_area=1.0)
    assert load_color_config(path, custom).blob is custom


@pytest.mark.software
def test_to_hsv_known_values():
    px = np.array([[PURE["red"], PURE["yellow"], PURE["green"], (255, 0, 0), (255, 255, 255)]], np.uint8)
    hsv = cb._to_hsv(px)
    assert hsv.dtype == np.uint8 and hsv.shape == px.shape
    assert hsv.reshape(-1, 3).tolist() == [[0, 255, 255], [30, 255, 255], [60, 255, 255], [120, 255, 255], [0, 0, 255]]


@pytest.mark.software
def test_red_threshold_wraps_around_the_hue_axis():
    hues = [0, 5, 10, 11, 90, 169, 170, 175, 179]
    hsv = np.array([[[h, 255, 255] for h in hues]], np.uint8)
    mask = cb._threshold_red(hsv, TEST_HSV)
    assert (mask[0] > 0).tolist() == [True, True, True, False, False, False, True, True, True]


@pytest.mark.software
def test_single_band_bounds_are_inclusive_and_saturation_and_value_gate():
    px = [(39, 255, 255), (40, 255, 255), (80, 255, 255), (81, 255, 255),      # hue edges
          (60, 119, 255), (60, 120, 255), (60, 255, 119), (60, 255, 120)]      # S and V edges
    mask = cb._threshold_single(np.array([px], np.uint8), TEST_HSV.green)
    assert (mask[0] > 0).tolist() == [False, True, True, False, False, True, False, True]


@pytest.mark.software
def test_mean_hsv_of_a_uniform_region():
    hsv = np.zeros((40, 40, 3), np.uint8)
    hsv[10:20, 5:25] = (30, 200, 180)
    contour = np.array([[5, 10], [24, 10], [24, 19], [5, 19]], np.int32).reshape(-1, 1, 2)
    assert cb._mean_hsv(hsv, contour) == (30.0, 200.0, 180.0)


SHAPE = (120, 240)


@pytest.mark.software
def test_accepted_blob_reports_its_exact_bbox_and_carries_identity():
    (c,), rc = blobs_of(rect_mask(SHAPE, (10, 5, 20, 20)), label="green")
    assert c.bbox == (10, 5, 20, 20) and c.label == "green"
    assert (c.frame_id, c.timestamp_ms) == (3, 4)
    assert rc == {"seen": 1, "area": 0, "aspect": 0, "round": 0, "core": 0, "accepted": 1}


@pytest.mark.software
@pytest.mark.parametrize("rect, gate", [
    ((5, 5, 6, 6), "area"),            # below min_area
    ((0, 0, 100, 100), "area"),        # above max_area
    ((0, 0, 90, 20), "aspect"),        # 4.5 wide
    ((0, 0, 20, 90), "aspect"),        # 0.22 tall
])
def test_each_blob_gate_rejects_and_is_counted_under_its_own_name(rect, gate):
    out, rc = blobs_of(rect_mask(SHAPE, rect))
    assert out == [] and rc[gate] == 1 and rc["accepted"] == 0
    assert rc["seen"] == rc["area"] + rc["aspect"] + rc["round"] + rc["core"] + rc["accepted"]


@pytest.mark.software
def test_blobs_exactly_at_the_area_limits_are_accepted():
    lo = BlobFilter(min_area=49.0, max_area=5000.0, min_aspect=0.3, max_aspect=3.0, ref_area=800.0, min_roundness=0.0, min_core_px=0)
    (c,), _ = blobs_of(rect_mask(SHAPE, (10, 10, 8, 8)), lo)          # contour area is (8-1)^2 = 49
    assert c.confidence == 0.0


@pytest.mark.software
def test_every_blob_in_a_mask_is_judged_independently():
    mask = rect_mask(SHAPE, (10, 5, 20, 20), (60, 5, 6, 6), (100, 5, 90, 20), (200, 40, 24, 24))
    out, rc = blobs_of(mask)
    assert sorted(c.bbox for c in out) == [(10, 5, 20, 20), (200, 40, 24, 24)]
    assert rc == {"seen": 4, "area": 1, "aspect": 1, "round": 0, "core": 0, "accepted": 2}


@pytest.mark.software
def test_reject_counts_argument_is_optional():
    assert len(cb._blobs_to_candidates(rect_mask(SHAPE, (10, 5, 20, 20)), "red", TEST_BLOB, 0, 0)) == 1


@pytest.mark.software
def test_confidence_is_area_only_monotonic_and_saturating():
    confs = [blobs_of(rect_mask((80, 80), (5, 5, k, k)))[0][0].confidence for k in range(9, 45, 3)]
    assert confs == sorted(confs) and confs[0] < 0.1 and confs[-1] == 1.0
    mask = rect_mask(SHAPE, (10, 10, 24, 24))
    assert len({blobs_of(mask, label=lb)[0][0].confidence for lb in LABELS}) == 1     # color plays no part


@pytest.mark.software
def test_trace_covers_accepted_and_rejected_blobs_with_the_documented_fields():
    flt = replace(TEST_BLOB, min_area=200.0)
    mask = rect_mask(SHAPE, (10, 5, 24, 24), (0, 60, 100, 3), (150, 60, 90, 20), (200, 5, 5, 5))
    trace = []
    _, rc = blobs_of(mask, flt, trace=trace)
    by_gate = {t["gate"]: t for t in trace}
    assert set(by_gate) == {None, "area", "aspect"}
    assert all(set(t) == TRACE_KEYS and t["label"] == "red" for t in trace)
    assert rc["seen"] == 4 and len(trace) == 3             # the 5x5 speckle is counted but not traced
    assert by_gate["area"]["aspect"] is None and by_gate["area"]["confidence"] is None
    assert by_gate["aspect"]["aspect"] == round(90 / 20, 3) and by_gate["aspect"]["confidence"] is not None
    assert by_gate[None]["fill"] == round(by_gate[None]["area"] / (24 * 24), 3)


@pytest.mark.software
def test_trace_mean_hsv_is_filled_only_when_an_hsv_image_is_given():
    mask = rect_mask(SHAPE, (10, 5, 24, 24))
    hsv = np.zeros(SHAPE + (3,), np.uint8); hsv[...] = (30, 200, 180)
    without, with_ = [], []
    blobs_of(mask, trace=without)
    blobs_of(mask, trace=with_, hsv=hsv)
    assert without[0]["hsv"] is None and with_[0]["hsv"] == (30.0, 200.0, 180.0)


@pytest.mark.software
def test_each_color_is_found_once_where_it_was_drawn():
    img = scene(blobs=THREE)
    cands, dbg = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB, 7, 8)
    assert sorted(c.label for c in cands) == sorted(LABELS)
    for label, cx, cy, r in THREE:
        (c,) = [c for c in cands if c.label == label]
        x, y, w, h = c.bbox
        assert abs(x + w / 2 - cx) <= 2 and abs(y + h / 2 - cy) <= 2              # rasterized circle is off-center by a pixel at most
        assert dbg["mask_px"][label] == pytest.approx(np.pi * r * r, rel=0.10)    # rasterization vs pi r^2 at r = 10
    assert_color_contract(cands, dbg, 7, 8)


@pytest.mark.software
@pytest.mark.parametrize("hue", [5, 175])
def test_red_is_found_on_both_sides_of_the_hue_wrap(hue):
    cands, _ = extract_traffic_light_candidates(scene(blobs=[(bgr_from_hsv(hue), 60, 60, 10)]), TEST_HSV, TEST_BLOB)
    assert [c.label for c in cands] == ["red"]


@pytest.mark.software
@pytest.mark.parametrize("bg, blobs", [
    (0, ()), (255, ()),                                              # black / white background
    (BG, [((255, 255, 255), 60, 60, 12)]),                           # white blob: no saturation
    (BG, [(bgr_from_hsv(90), 60, 60, 12)]),                          # cyan: outside every band
    (BG, [(bgr_from_hsv(0, 90, 255), 60, 60, 12)]),                  # washed-out red: below the S floor
])
def test_scenes_without_a_saturated_in_band_blob_yield_nothing(bg, blobs):
    img = scene(blobs=blobs); img[:6, :6] = bg
    cands, dbg = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB)
    assert cands == [] and dbg["mask_px"] == {"red": 0, "yellow": 0, "green": 0}


@pytest.mark.software
def test_moving_one_hsv_band_removes_only_that_color():
    moved = replace(TEST_HSV, green=ColorRange((100, 120, 120), (120, 255, 255)))
    cands, _ = extract_traffic_light_candidates(scene(blobs=THREE), moved, TEST_BLOB)
    assert sorted(c.label for c in cands) == ["red", "yellow"]


@pytest.mark.software
def test_blob_filter_reaches_the_gates_without_touching_the_masks():
    img = scene(blobs=THREE)
    _, base = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB)
    cands, dbg = extract_traffic_light_candidates(img, TEST_HSV, replace(TEST_BLOB, min_area=1e6))
    assert cands == [] and dbg["mask_px"] == base["mask_px"]
    assert all(dbg["reject_counts"][c]["area"] == 1 for c in LABELS)


@pytest.mark.software
def test_small_blob_and_streak_are_rejected_at_their_gates():
    img = scene(blobs=[("red", 20, 30, 3)])
    cv2.rectangle(img, (60, 20), (119, 25), PURE["green"], -1)
    cands, dbg = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB, trace=True)
    assert cands == []
    assert dbg["reject_counts"]["red"]["area"] == 1 and dbg["reject_counts"]["green"]["aspect"] == 1
    assert [(t["label"], t["gate"]) for t in dbg["trace"]] == [("green", "aspect")]


@pytest.mark.software
def test_debug_dict_shape_and_flags():
    img = scene(blobs=THREE)
    _, dbg = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB)
    assert dbg["roi"] is img and dbg["calibrated"] is False and "trace" not in dbg
    assert set(dbg) >= {"hsv", "red", "yellow", "green", "roi", "mask_px", "reject_counts", "calibrated"}


@pytest.mark.software
def test_loaded_ranges_report_calibrated(tmp_path):
    ranges = load_hsv_ranges(write_json(tmp_path, hsv_json()))
    _, dbg = extract_traffic_light_candidates(scene(blobs=THREE), ranges, TEST_BLOB)
    assert dbg["calibrated"] is True


@pytest.mark.software
def test_uncalibrated_scaffold_ranges_are_accepted_for_tuning():
    cands, dbg = extract_traffic_light_candidates(scene(blobs=THREE), HSVRanges(), TEST_BLOB)
    assert dbg["calibrated"] is False and {c.label for c in cands} == set(LABELS)


@pytest.mark.software
def test_extraction_is_deterministic_read_only_safe_and_leaves_the_roi_alone():
    img = scene(blobs=THREE)
    before = img.copy()
    img.flags.writeable = False                                       # crop_rois hands out read-only views
    a, _ = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB, 1, 2)
    b, _ = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB, 1, 2)
    assert np.array_equal(img, before)
    assert [(c.label, c.bbox, c.confidence) for c in a] == [(c.label, c.bbox, c.confidence) for c in b]


@pytest.mark.software
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (1, 50), (50, 1)])
def test_tiny_rois_do_not_raise(shape):
    roi = np.random.default_rng(0).integers(0, 256, shape + (3,), dtype=np.uint8)
    extract_traffic_light_candidates(roi, TEST_HSV, TEST_BLOB, trace=True)


@pytest.mark.software
@pytest.mark.parametrize("seed", range(6))
def test_invariants_hold_on_cluttered_rois(seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 80, (120, 240, 3), dtype=np.uint8)
    for _ in range(25):
        cv2.circle(img, (int(rng.integers(0, 240)), int(rng.integers(0, 120))), int(rng.integers(2, 25)),
                   bgr_from_hsv(int(rng.integers(0, 180)), int(rng.integers(150, 256)), int(rng.integers(150, 256))), -1)
    cands, dbg = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB, 9, 8, trace=True)
    assert_color_contract(cands, dbg, 9, 8)
    assert all(t["gate"] in (None, "area", "aspect") for t in dbg["trace"])


@pytest.mark.software
def test_trace_is_only_present_when_requested():
    img = scene(blobs=THREE)
    _, off = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB, trace=False)
    _, on = extract_traffic_light_candidates(img, TEST_HSV, TEST_BLOB, trace=True)
    assert "trace" not in off and [t["gate"] for t in on["trace"]] == [None, None, None]


@pytest.mark.software
@pytest.mark.parametrize("bad, exc", [
    (None, ValueError),
    (np.zeros((20, 20, 3), np.float32), TypeError),
    (np.zeros((20, 20), np.uint8), ValueError),                       # gray has no chroma
    (np.zeros((20, 20, 4), np.uint8), ValueError),
])
def test_invalid_rois_are_rejected(bad, exc):
    with pytest.raises(exc, match="extract_traffic_light_candidates"):
        extract_traffic_light_candidates(bad, TEST_HSV, TEST_BLOB)


@pytest.mark.software
def test_missing_hsv_ranges_are_rejected_with_a_pointer_to_the_loader():
    with pytest.raises(ValueError, match="load_hsv_ranges"):
        extract_traffic_light_candidates(scene(), None, TEST_BLOB)


@pytest.mark.software
def test_stage_is_off_without_ranges_and_never_touches_the_image():
    roi, _ = build_traffic_scene(ROIConfig())
    assert run_color_stage(roi, ColorConfig()) == ([], {"enabled": False})


@pytest.mark.software
@pytest.mark.parametrize("roi_cfg", ROI_CONFIGS.values(), ids=ROI_CONFIGS.keys())
def test_chain_finds_each_color_at_its_frame_coordinates(roi_cfg):
    roi, expect = build_traffic_scene(roi_cfg)
    assert not roi.traffic_roi.flags.writeable                        # read-only view must be accepted
    cands, dbg = run_color_stage(roi, TEST_CFG)
    assert dbg["enabled"] is True
    assert_color_contract(cands, dbg, roi.frame_id, roi.timestamp_ms)
    assert (roi.frame_id, roi.timestamp_ms) == (11, 222)
    assert sorted(c.label for c in cands) == sorted(LABELS)
    tx, ty = roi.traffic_rect[:2]
    for c in cands:                                                   # ROI-relative -> add the rect origin
        x, y, w, h = c.bbox
        ex, ey = expect[c.label]
        assert abs(tx + x + w / 2 - ex) <= 3 and abs(ty + y + h / 2 - ey) <= 3   # wider than direct: preprocess blurs the edges


@pytest.mark.software
def test_stage_config_components_reach_the_extractor():
    roi, _ = build_traffic_scene(ROIConfig())
    strict, _ = run_color_stage(roi, ColorConfig(TEST_HSV, replace(TEST_BLOB, min_area=1e9)))
    moved, _ = run_color_stage(roi, ColorConfig(replace(TEST_HSV, red_low=ColorRange((100, 120, 120), (110, 255, 255)),
                                                                red_high=ColorRange((111, 120, 120), (120, 255, 255))), TEST_BLOB))
    assert strict == [] and sorted(c.label for c in moved) == ["green", "yellow"]


@pytest.mark.software
def test_stage_trace_is_passed_through():
    roi, _ = build_traffic_scene(ROIConfig())
    _, off = run_color_stage(roi, TEST_CFG)
    _, on = run_color_stage(roi, TEST_CFG, trace=True)
    assert "trace" not in off and len(on["trace"]) == 3


@pytest.mark.software
@pytest.mark.parametrize("roi_cfg", ROI_CONFIGS.values(), ids=ROI_CONFIGS.keys())
def test_stage_analyzes_the_traffic_roi_and_adds_nothing(roi_cfg):
    # The stage is the extractor plus the frame stamp: boxes stay ROI-relative,
    # the array analyzed is roi.traffic_roi, and nothing is re-cropped or offset.
    roi, _ = build_traffic_scene(roi_cfg)
    crop = roi.traffic_roi
    h, w = roi.traffic_rect[3], roi.traffic_rect[2]

    cands, dbg = run_color_stage(roi, TEST_CFG)
    assert np.array_equal(dbg["roi"], crop) and dbg["roi"].shape[:2] == (h, w)
    assert len(cands) == 3

    direct, _ = extract_traffic_light_candidates(crop, TEST_HSV, TEST_BLOB, roi.frame_id, roi.timestamp_ms)
    key = lambda cs: sorted((c.label, c.bbox, c.confidence, c.frame_id, c.timestamp_ms) for c in cs)
    assert key(cands) == key(direct)

    blank, _ = run_color_stage(replace(roi, traffic_roi=np.zeros_like(crop)), TEST_CFG)
    assert blank == []                                                # the pixels come from roi.traffic_roi


@pytest.mark.software
@pytest.mark.parametrize("label, color", [("red", (0, 0, 255)), ("yellow", (0, 200, 255)),
                                          ("green", (0, 200, 0)), ("mystery", (255, 255, 255))])
def test_draw_candidates_marks_the_bbox_in_the_label_color(label, color):
    roi = np.zeros((80, 120, 3), np.uint8)
    cand = TrafficLightCandidate(label, (20, 30, 24, 24), 0.5, 1, 2)
    vis = draw_candidates(roi, [cand])
    assert tuple(vis[30, 20]) == color and tuple(vis[30 + 23, 20 + 23]) == color
    assert not np.array_equal(vis, roi)


@pytest.mark.software
def test_draw_candidates_returns_a_copy_and_leaves_the_input_alone():
    roi = scene(blobs=THREE)
    before = roi.copy()
    vis = draw_candidates(roi, [TrafficLightCandidate("red", (30, 50, 21, 21), 0.3, 1, 2)])
    assert np.array_equal(roi, before) and vis is not roi and not np.shares_memory(vis, roi)
    empty = draw_candidates(roi, [])
    assert np.array_equal(empty, roi) and empty is not roi


@pytest.mark.software
def test_every_recorded_frame_meets_the_color_contract(dataset_frames):
    if not dataset_frames:
        pytest.skip("no recorded dataset in tests/data/frames (run: pytest --hardware --record)")
    cfg = ColorConfig(HSVRanges(), BlobFilter())                      # scaffold: the contract, not the tuning
    for fd in dataset_frames:
        roi = crop_rois(preprocess_frame(fd))
        cands, dbg = run_color_stage(roi, cfg)
        try:
            assert_color_contract(cands, dbg, roi.frame_id, roi.timestamp_ms)
        except AssertionError as e:
            raise AssertionError(f"frame_id={fd.frame_id}: {e}") from e


def _hardware_config():
    """Calibrated ranges if the calibration file exists, else the scaffold (and the run says so)."""
    if CALIB.exists():
        return load_color_config(str(CALIB)), str(CALIB)
    return ColorConfig(HSVRanges(), BlobFilter()), "scaffold (uncalibrated)"


def _jsonable_trace(trace):
    """Blob trace with tuples converted to lists for JSON."""
    return [{**t, "bbox": list(t["bbox"]), "hsv": None if t["hsv"] is None else list(t["hsv"])} for t in trace]


@pytest.mark.hardware
def test_color_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    cfg, source = _hardware_config()

    rows, samples = [], {}
    gate_totals = {c: {g: 0 for g in GATES} for c in LABELS}
    for i, fd in enumerate(frames(n)):
        roi = crop_rois(preprocess_frame(fd))                         # upstream, not timed

        t0 = time.perf_counter_ns()
        cands, dbg = run_color_stage(roi, cfg)      # live-loop settings: no trace
        stage_ms = (time.perf_counter_ns() - t0) / 1e6

        assert_color_contract(cands, dbg, roi.frame_id, roi.timestamp_ms)   # outside the timing window
        for c in LABELS:
            for g in GATES:
                gate_totals[c][g] += dbg["reject_counts"][c][g]
        rows.append((fd.frame_id, fd.timestamp_ms, round(stage_ms, 3), len(cands),
                     *(sum(1 for c in cands if c.label == lb) for lb in LABELS),
                     *(dbg["mask_px"][lb] for lb in LABELS),
                     *(dbg["reject_counts"][lb][g] for lb in LABELS for g in GATES)))
        if i in (0, n // 2, n - 1):
            samples[fd.frame_id] = roi

    if not rows:
        pytest.skip("no frames delivered")

    header = (["frame_id", "timestamp_ms", "stage_ms", "n_candidates"] + [f"n_{lb}" for lb in LABELS]
              + [f"mask_px_{lb}" for lb in LABELS] + [f"{lb}_{g}" for lb in LABELS for g in GATES])
    stage = [r[2] for r in rows]
    artifacts.json("config.json", {"hsv_source": source, "calibrated": cfg.hsv_ranges.is_calibrated,
                                   "ranges": asdict(cfg.hsv_ranges), "blob": asdict(cfg.blob)})
    artifacts.csv("color_timing.csv", header, rows)
    artifacts.json("summary.json", {
        "hsv_source": source, "calibrated": cfg.hsv_ranges.is_calibrated,
        "stage_ms": summarize(stage), "frames_over_33ms": sum(1 for s in stage if s > 33.3),   # one frame period at MAX_FPS (30)
        "frames_with_candidate": {lb: sum(1 for r in rows if r[4 + LABELS.index(lb)]) for lb in LABELS},
        "mean_mask_px": {lb: float(np.mean([r[7 + LABELS.index(lb)] for r in rows])) for lb in LABELS},
        "gate_totals": gate_totals,
    })
    artifacts.histogram("stage_ms_hist.png", stage, "run_color_stage latency", "ms")

    for fid, roi in samples.items():                         # untimed, with the debug views
        cands, dbg = run_color_stage(roi, cfg, trace=True)
        artifacts.image(f"{fid:06d}_traffic_roi.png", dbg["roi"])
        for lb in LABELS:
            artifacts.image(f"{fid:06d}_mask_{lb}.png", dbg[lb])
        artifacts.image(f"{fid:06d}_overlay.png", draw_candidates(dbg["roi"], cands))
        artifacts.json(f"{fid:06d}_blob_trace.json", _jsonable_trace(dbg["trace"]))


# =============================================================================
# A lamp or a shirt: the roundness and core gates, at the robot's BlobFilter()
# =============================================================================

PROD = BlobFilter()                         # every gate on, as the robot runs
LAMP_BOX = (120, 200)                       # (h, w) of the drawn traffic ROI


def lit(color, r=12, core_r=4, core=(250, 250, 250), center=(100, 60), shape=None):
    """A lamp as the camera records it: a colored disc with a clipped near-white middle (or a given shape)."""
    img = np.zeros((*LAMP_BOX, 3), np.uint8)
    if shape is None:
        cv2.circle(img, center, r, PURE[color], -1)
    else:
        shape(img, PURE[color])
    if core_r:
        cv2.circle(img, center, core_r, core, -1)
    return img


def found(img, blob=PROD):
    cands, dbg = extract_traffic_light_candidates(img, TEST_HSV, blob, trace=True)
    return [c.label for c in cands], dbg


@pytest.mark.software
@pytest.mark.parametrize("color", LABELS)
def test_a_lit_lamp_of_each_color_passes_every_gate(color):
    labels, dbg = found(lit(color))
    assert labels == [color]
    (t,) = [t for t in dbg["trace"] if t["gate"] is None]
    assert t["roundness"] > 0.8 and t["core_px"] >= PROD.min_core_px


@pytest.mark.software
@pytest.mark.parametrize("color", LABELS)
@pytest.mark.parametrize("r", [8, 12, 16])
def test_a_shirt_the_lamps_color_and_size_is_rejected_for_having_no_clipped_core(color, r):
    labels, dbg = found(lit(color, r=r, core_r=0))
    assert labels == [] and dbg["reject_counts"][color]["core"] == 1


@pytest.mark.software
def test_a_bar_the_aspect_gate_passes_is_rejected_as_not_round():
    bar = lambda img, bgr: cv2.rectangle(img, (75, 50), (125, 70), bgr, -1)       # 2.5:1, inside max_aspect 3
    labels, dbg = found(lit("red", shape=bar))
    assert labels == [] and dbg["reject_counts"]["red"]["round"] == 1
    (t,) = dbg["trace"]
    assert t["gate"] == "round" and t["roundness"] < PROD.min_roundness and t["aspect"] <= PROD.max_aspect


@pytest.mark.software
@pytest.mark.parametrize("core, core_r, passes", [
    ((250, 250, 250), 4, True),
    ((235, 235, 235), 4, False),            # V 235: bright, but under core_min_v (240): not clipped
    ((140, 160, 250), 4, False),            # V 250 but S ~112, over core_max_s (60): colored, not white
    ((250, 250, 250), 0, False),            # no core at all
])
def test_only_a_clipped_near_white_core_counts(core, core_r, passes):
    labels, _ = found(lit("green", core=core, core_r=core_r))
    assert (labels == ["green"]) is passes


@pytest.mark.software
def test_the_core_needs_min_core_px_pixels():
    img = lit("yellow", core_r=0)
    img[60, 100] = img[60, 101] = (250, 250, 250)                   # 2 clipped px
    assert found(img)[0] == [] and found(img, replace(PROD, min_core_px=2))[0] == ["yellow"]


@pytest.mark.software
def test_with_the_core_gate_off_a_solid_disc_passes_and_without_hsv_the_gate_refuses():
    assert found(lit("red", core_r=0), replace(PROD, min_core_px=0))[0] == ["red"]
    mask = np.zeros(LAMP_BOX, np.uint8)
    cv2.circle(mask, (100, 60), 12, 255, -1)
    with pytest.raises(ValueError, match="core gate needs"):
        cb._blobs_to_candidates(mask, "red", PROD, 0, 0)
    with pytest.raises(ValueError, match="core gate needs"):
        cb._filter_blobs(mask, "red", PROD, 0, 0)


@pytest.mark.software
def test_the_production_twin_takes_lamps_and_rejects_shirts_alike():
    for img in (lit("red"), lit("green", core_r=0), lit("yellow", r=16)):
        a, _ = extract_traffic_light_candidates(img, TEST_HSV, PROD, 3, 4)
        b = cb.find_traffic_light_candidates(img, TEST_HSV, PROD, 3, 4)
        assert a == b


@pytest.mark.software
def test_clipped_pixels_beside_a_shirt_dont_count_as_its_core():
    """White inside the shirt's bounding box but outside its outline: a window or a white sleeve, not a lamp's core."""
    img = lit("red", core_r=0)                                      # disc r 12 at (100, 60): its box is 88-112 x 48-72
    img[48:51, 88:91] = (250, 250, 250)                             # the box's corner, outside the circle
    labels, dbg = found(img)
    assert labels == [] and dbg["reject_counts"]["red"]["core"] == 1



# =============================================================================
# Glow mode: the white spot with the most white pixels, named by its ring
# =============================================================================

GLOW = GlowFilter()


def led(roi, center, color, white_r=2, ring_r=4):
    """A lit LED: a colored ring around a clipped white center (BGR roi, drawn in place)."""
    cv2.circle(roi, center, ring_r, PURE[color], -1)
    cv2.circle(roi, center, white_r, (255, 255, 255), -1)


def glow_roi():
    return np.full((60, 120, 3), BG, np.uint8)


def glow_read(roi, glow=GLOW, trace=False):
    return extract_traffic_light_candidates(roi, TEST_HSV, TEST_BLOB, 7, 70, trace=trace, glow=glow)


@pytest.mark.software
def test_glow_finds_a_lit_led_by_its_white_center_and_names_it_by_its_ring():
    roi = glow_roi()
    led(roi, (40, 30), "red")
    cands, dbg = glow_read(roi)
    assert [(c.label, c.confidence, c.frame_id, c.timestamp_ms) for c in cands] == [("red", 1.0, 7, 70)]
    x, y, w, h = cands[0].bbox
    assert x <= 38 and x + w >= 42 and y <= 28 and y + h >= 32          # the white center and its ring
    assert dbg["reject_counts"]["red"] == {"seen": 1, "white": 0, "shape": 0, "smaller": 0, "accepted": 1}


@pytest.mark.software
def test_glow_never_picks_an_unlit_lens_however_large_or_bright():
    roi = glow_roi()
    cv2.circle(roi, (90, 30), 14, PURE["green"], -1)                    # big, saturated, V 255: no white center
    led(roi, (30, 30), "yellow")
    cands, _ = glow_read(roi)
    assert [c.label for c in cands] == ["yellow"]
    unlit_only = glow_roi()
    cv2.circle(unlit_only, (90, 30), 14, PURE["green"], -1)
    assert glow_read(unlit_only)[0] == []


@pytest.mark.software
def test_glow_keeps_the_spot_with_the_most_white_over_a_reflection():
    roi = glow_roi()
    led(roi, (30, 40), "green", white_r=3, ring_r=5)
    led(roi, (90, 15), "yellow", white_r=1, ring_r=3)                   # the dimmer reflection, read orange
    cands, dbg = glow_read(roi, trace=True)
    assert [c.label for c in cands] == ["green"]
    gates = {e["label"]: e["gate"] for e in dbg["trace"]}
    assert gates == {"green": None, "yellow": "smaller"}
    assert all(set(e) == TRACE_KEYS for e in dbg["trace"])


@pytest.mark.software
def test_glow_a_white_spot_without_a_colored_ring_or_too_small_is_no_light():
    roi = glow_roi()
    cv2.circle(roi, (30, 30), 3, (255, 255, 255), -1)                   # glare: white, no ring color
    roi[10, 90] = (255, 255, 255)                                       # one clipped pixel
    cv2.circle(roi, (90, 10), 3, PURE["red"], -1)
    roi[10, 90] = (255, 255, 255)
    cands, dbg = glow_read(roi, trace=True)
    assert cands == []
    assert sorted((e["label"], e["gate"]) for e in dbg["trace"]) == [("none", "ring"), ("red", "white")]
    assert dbg["reject_counts"]["red"]["white"] == 1


@pytest.mark.software
def test_glow_confidence_is_white_pixels_over_ref_white_px():
    roi = glow_roi()
    roi[29:31, 39:41] = 255
    cv2.circle(roi, (40, 30), 4, PURE["red"], -1)
    roi[29:31, 39:41] = 255                                             # 4 white px
    cands, _ = glow_read(roi)
    assert cands[0].confidence == round(4 / GLOW.ref_white_px, 4)
    assert glow_read(roi, replace(GLOW, ref_white_px=2.0))[0][0].confidence == 1.0
    assert glow_read(roi, replace(GLOW, min_white_px=5))[0] == []


@pytest.mark.software
def test_glow_twins_and_the_stage_agree():
    roi = glow_roi()
    led(roi, (40, 30), "green")
    led(roi, (90, 30), "red", white_r=1)
    cands, _ = glow_read(roi)
    assert find_traffic_light_candidates(roi, TEST_HSV, TEST_BLOB, 7, 70, glow=GLOW) == cands
    frame = np.full((FRAME_H, FRAME_W, 3), BG, np.uint8)
    probe = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig())
    tx, ty, tw, th = probe.traffic_rect
    led(frame, (tx + tw // 2, ty + th // 2), "green", white_r=3, ring_r=6)
    roi_res = crop_rois(preprocess_frame(FrameData(frame, 3, 30)), ROIConfig())
    cfg = ColorConfig(TEST_HSV, TEST_BLOB, GLOW)
    staged, dbg = run_color_stage(roi_res, cfg)
    assert [c.label for c in staged] == ["green"] and dbg["enabled"]
    assert detect_color(roi_res, cfg) == staged
    assert [c.label for c in run_color_stage(roi_res, ColorConfig(TEST_HSV, TEST_BLOB))[0]] == ["green"]   # blob mode too


@pytest.mark.software
def test_load_color_config_carries_glow_and_leaves_it_off_by_default():
    assert load_color_config(str(CALIB)).glow is None
    assert load_color_config(str(CALIB), glow=GLOW).glow == GLOW


@pytest.mark.software
def test_glow_names_a_spot_by_its_ring_not_its_own_tinted_white():
    # A clipped center keeps a trace of tint (S up to white_max_s); with a band
    # whose S floor is under that, as yellow's 20 is on the robot, the center
    # itself would vote for that band
    hsv = replace(TEST_HSV, yellow=ColorRange((20, 20, 120), (35, 255, 255)))
    roi = glow_roi()
    cv2.circle(roi, (40, 30), 4, PURE["red"], -1)
    cv2.circle(roi, (40, 30), 3, bgr_from_hsv(28, 50, 255), -1)       # white with a yellow tint, 29 px
    cands, _ = extract_traffic_light_candidates(roi, hsv, TEST_BLOB, glow=GLOW)
    assert [c.label for c in cands] == ["red"]



@pytest.mark.software
def test_glow_a_flat_strip_of_glare_is_no_lamp_however_white_or_ringed():
    # The board's edge glare (2026-10-07): a 2 px tall strip, ~40 white px,
    # more than the lit LED's, whose ring picked up a red shirt below
    roi = glow_roi()
    cv2.rectangle(roi, (10, 48), (40, 52), PURE["red"], -1)
    cv2.rectangle(roi, (12, 50), (38, 51), (255, 255, 255), -1)        # 27 x 2 white
    led(roi, (80, 20), "green", white_r=2)
    cands, dbg = glow_read(roi, trace=True)
    assert [c.label for c in cands] == ["green"]
    assert ("red", "shape") in [(e["label"], e["gate"]) for e in dbg["trace"]]
    assert dbg["reject_counts"]["red"]["shape"] == 1
    tall = glow_roi()
    cv2.rectangle(tall, (38, 5), (42, 55), PURE["red"], -1)
    cv2.rectangle(tall, (40, 8), (40, 52), (255, 255, 255), -1)         # 1 x 45 white
    assert glow_read(tall)[0] == []
    assert [c.label for c in glow_read(roi, replace(GLOW, max_white_aspect=20.0))[0]] == ["red"]



@pytest.mark.software
def test_glow_a_white_spot_needs_its_color_on_a_share_of_its_ring_not_a_few_stray_pixels():
    # A glare (2026-10-07): a big white spot whose ring touched a few colored
    # pixels, past min_ring_px but ~5% of the ring
    roi = glow_roi()
    cv2.rectangle(roi, (20, 20), (31, 31), (255, 255, 255), -1)       # 12 x 12 white, a ~56 px ring
    roi[18, 24:28] = PURE["red"]                                       # 4 stray red px in the ring
    cands, dbg = glow_read(roi, trace=True)
    assert cands == [] and [(e["label"], e["gate"]) for e in dbg["trace"]] == [("none", "ring")]
    assert [c.label for c in glow_read(roi, replace(GLOW, min_ring_share=0.0))[0]] == ["red"]
    lit = glow_roi()
    led(lit, (60, 30), "red", white_r=4, ring_r=6)                      # its ring colored all round
    assert [c.label for c in glow_read(lit)[0]] == ["red"]


@pytest.mark.software
def test_glow_a_small_spot_still_needs_min_ring_px_of_color():
    roi = glow_roi()
    roi[30, 40:42] = (255, 255, 255)                                   # 2 white px, a ~14 px ring
    roi[29, 40:42] = PURE["green"]                                     # 2 colored: over 10%, under 3 px
    assert glow_read(roi)[0] == []
    roi[31, 40] = PURE["green"]                                        # a third
    assert [c.label for c in glow_read(roi)[0]] == ["green"]
