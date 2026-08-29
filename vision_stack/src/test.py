"""Bench test for fixes. Not part of the pipeline."""
import numpy as np
import cv2

from config import Config, FrameTags, intersection_edge_ratio
from roi_crop import crop_rois
from geometry import extract_lane_candidates, CannyParams, LaneContourFilter
from lane_offset import compute_lane_offset
from feature_fusion import DetectionObject

FAIL = 0
def check(name, cond):
    global FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond: FAIL += 1

# ---------------------------------------------------------------- Fix 1
print("Fix 1: roi_inset")
frame = np.zeros((360, 480, 3), dtype=np.uint8)
base   = crop_rois(frame, 0)                              # baseline (no cfg)
alloff = crop_rois(frame, 0, fix_cfg=Config())         # all flags False
fix1   = crop_rois(frame, 0, fix_cfg=Config(roi_inset=True))
check("None == all-off (lane_rect)", base.lane_rect == alloff.lane_rect == (0, 180, 480, 180))
check("traffic/sign untouched", fix1.traffic_rect == base.traffic_rect and fix1.sign_rect == base.sign_rect)
check("inset lane_rect = (57, 216, 366, 144)", fix1.lane_rect == (57, 216, 366, 144))
lx, ly, lw, lh = fix1.lane_rect
check("symmetric: ROI center == frame center", lx + lw / 2 == 480 / 2)
check("roi view shape matches rect", fix1.lane_roi.shape[:2] == (lh, lw))

# ---------------------------------------------------------------- synthetic lane ROI
# 180x480 dark ROI with bright markings. Filter tuned so the test shapes are
# comfortably inside acceptance (the stock filter is course-calibrated).
def make_roi():
    roi_bgr = np.full((180, 480, 3), 30, dtype=np.uint8)
    return roi_bgr

lf = LaneContourFilter(min_area=0.0, max_area=100000.0, min_aspect=4.0,
                       max_aspect=30.0, ref_area=2000.0, max_roi_span=1.0,
                       min_intensity=80.0)
cp = CannyParams()

# ---------------------------------------------------------------- Fix 3
print("Fix 3: orientation_filt")
roi = make_roi()
cv2.rectangle(roi, (100, 140), (180, 152), (255, 255, 255), -1)  # horizontal line, low
cv2.rectangle(roi, (300, 10), (310, 120), (255, 255, 255), -1)   # steep pole, high
roi_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)  # _to_grayscale expects YUV-ish input

tags_off = FrameTags()
cand_base, _ = extract_lane_candidates(roi_yuv, cp, lf, 0, 0, tags=tags_off)
cand_none, _ = extract_lane_candidates(roi_yuv, cp, lf, 0, 0,)  # no cfg no tags
check("baseline: both shapes accepted", len(cand_base) == 2 and len(cand_none) == 2)
check("tags-only run doesn't change acceptance", len(cand_base) == len(cand_none))
check("pole tagged with fix OFF", tags_off.pole_misclassified >= 1)

tags_on = FrameTags()
cand_f3, _ = extract_lane_candidates(roi_yuv, cp, lf, 0, 0,
                                     fix_cfg=Config(orientation_filt=True), tags=tags_on)
check("fix ON: pole rejected, horizontal kept", len(cand_f3) == 1)
check("pole tag disappears with fix ON", tags_on.pole_misclassified == 0)
kept = cand_f3[0]
check("kept candidate is the horizontal one", kept.bbox[1] > 90)

# ---------------------------------------------------------------- Fix 2
print("Fix 2: trapezoid_mask")
roi = make_roi()
cv2.rectangle(roi, (2, 20), (60, 32), (255, 255, 255), -1)      # far-left top corner (outside trapezoid)
cv2.rectangle(roi, (200, 150), (290, 162), (255, 255, 255), -1) # bottom-center (inside)
roi_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
cand_b, dbg_b = extract_lane_candidates(roi_yuv, cp, lf, 0, 0)
cand_m, dbg_m = extract_lane_candidates(roi_yuv, cp, lf, 0, 0,
                                        fix_cfg=Config(trapezoid_mask=True))
check("baseline: 2 candidates", len(cand_b) == 2)
check("mask ON: corner candidate removed", len(cand_m) == 1)
check("edges_processed zeroed outside trapezoid", np.count_nonzero(dbg_m["edges_processed"][:35, :65]) == 0)
check("edges key unchanged (raw canny preserved)", np.array_equal(dbg_m["edges"], dbg_b["edges"]))

# ---------------------------------------------------------------- Fix 4
print("Fix 4: dashed_dilate")
roi = make_roi()
for y0 in (60, 90, 120, 150):                                   # 4 vertical dashes, 12px gaps
    cv2.rectangle(roi, (238, y0), (244, y0 + 18), (255, 255, 255), -1)
roi_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
_, dbg_b4 = extract_lane_candidates(roi_yuv, cp, lf, 0, 0)
_, dbg_d4 = extract_lane_candidates(roi_yuv, cp, lf, 0, 0,
                                    fix_cfg=Config(dashed_dilate=True))
n_b = len(cv2.findContours(dbg_b4["edges_processed"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
n_d = len(cv2.findContours(dbg_d4["edges_processed"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
check(f"dilate merges dash contours ({n_b} -> {n_d})", n_d < n_b and n_d == 1)

# dashed_reject_center tagging: tiny center fragments rejected for area
tags_d = FrameTags()
lf_strict = LaneContourFilter(min_area=500.0, max_area=100000.0, min_aspect=1.0,
                              max_aspect=30.0, ref_area=2000.0, max_roi_span=1.0,
                              min_intensity=80.0)
extract_lane_candidates(roi_yuv, cp, lf_strict, 0, 0, tags=tags_d)
check("center-region rejections counted", tags_d.dashed_reject_center >= 3)

# ---------------------------------------------------------------- Fix 5
print("Fix 5: anchor_halves")
def det(x, conf=0.8):
    return DetectionObject(type="lane_boundary", position={"x": float(x), "y": 100.0},
                           confidence=conf, timestamp=0, bounding_box=(0, 0, 1, 1),
                           label_detail="lane_boundary")
W = 480  # frame_center = 240
same_half = [det(50), det(200)]   # both left of center, 150px apart (>= min width)
r_base = compute_lane_offset(same_half, W, 0, 0, tags=FrameTags())
tags5 = FrameTags()
r_tag  = compute_lane_offset(same_half, W, 0, 0, tags=tags5)
r_fix  = compute_lane_offset(same_half, W, 0, 0, fix_cfg=Config(anchor_halves=True), tags=FrameTags())
check("baseline: same-half pair -> two_boundary", r_base.mode == "two_boundary")
check("anchor_wrong_half tagged with fix OFF", tags5.anchor_wrong_half)
check("fix ON: downgrades to left_only", r_fix.mode == "left_only")
check("fix ON: anchor is leftmost of left pool", r_fix.left_x == 50.0)

normal = [det(60), det(420)]
rn_base = compute_lane_offset(normal, W, 0, 0)
rn_fix  = compute_lane_offset(normal, W, 0, 0, fix_cfg=Config(anchor_halves=True))
check("valid pair: fix ON == baseline offset", rn_base.offset == rn_fix.offset and rn_fix.mode == "two_boundary")
single = [det(100)]
rs_base = compute_lane_offset(single, W, 0, 0)
rs_fix  = compute_lane_offset(single, W, 0, 0, fix_cfg=Config(anchor_halves=True))
check("single boundary: fix ON == baseline", rs_base == rs_fix and rs_base.mode == "left_only")
r_none = compute_lane_offset([], W, 0, 0, fix_cfg=Config(anchor_halves=True))
check("empty detections: mode none", r_none.mode == "none")

# ---------------------------------------------------------------- intersection ratio
print("Intersection edge ratio")
edges = np.zeros((180, 480), dtype=np.uint8)
cv2.line(edges, (0, 160), (479, 160), 255, 2)   # stop-line-like horizontal band
cv2.line(edges, (100, 10), (110, 100), 255, 1)  # sparse upper edges
ratio_line = intersection_edge_ratio(edges)
uniform = np.zeros((180, 480), dtype=np.uint8)
uniform[::10, :] = 255
ratio_unif = intersection_edge_ratio(uniform)
check(f"stop line -> high ratio ({ratio_line:.2f})", ratio_line > 2.0)
check(f"uniform edges -> ~1.0 ({ratio_unif:.2f})", 0.7 < ratio_unif < 1.4)
check("empty ROI -> 0.0", intersection_edge_ratio(np.zeros((180, 480), np.uint8)) == 0.0)

# ---------------------------------------------------------------- flags/CLI plumbing
print("Config plumbing")
cfg = Config.from_names(["roi_inset", "anchor_halves"])
check("from_names sets flags", cfg.roi_inset and cfg.anchor_halves and not cfg.trapezoid_mask)
check("flags_str format", cfg.flags_str() == "I1 T0 O0 D0 A1")
try:
    Config.from_names(["bogus"]); check("unknown name raises", False)
except ValueError:
    check("unknown name raises", True)

print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
raise SystemExit(1 if FAIL else 0)
