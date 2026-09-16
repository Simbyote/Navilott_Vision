"""
feature_fusion.py

Feature Fusion Stage

Purpose:
    The color branch and geometry branch run independently and produce
    candidates in their own ROI coordinate spaces. Fusion has three
    responsibilities:

    1. Normalize representation:
        convert all branch-specific candidate types into the single Detection
        Object schema defined by Phase 2

    2. Per-class conflict resolution:
        each detection class could produce multiple candidates in a single
        frame. Fusion resolves conflicts between candidates of the same class
        and logs every discard or suppression in the debug summary

    3. Assign position:
        the centroid of the winning candidate's bounding box

Coordinate spaces:
    position and bounding_box are ROI-LOCAL, in the pixel coordinates of
    whichever ROI the candidate came from. A lane_boundary at (200, 50) and a
    stop_sign at (200, 50) are not the same place in the frame.

    Each DetectionObject therefore names its own space: source_roi identifies
    which ROI ("lane", "traffic", "sign") and source_rect is that ROI's
    (x, y, w, h) in frame coordinates. A consumer that wants frame
    coordinates adds the rect origin:

        frame_x = detection.position["x"] + detection.source_rect[0]
        frame_y = detection.position["y"] + detection.source_rect[1]

    Carrying the rect on the detection rather than the envelope keeps each
    detection self-describing once it crosses into navigation, where the
    ROICropResult is no longer in scope.

Frame identity:
    frame_id and timestamp_ms come from the ROICropResult, the common ancestor
    of both branches, so every detection fused from one frame carries the same
    stamp. Nothing here re-derives either value from a candidate or a clock.

Notes:
    bounding_box is retained as an internal debug field and is used only for
    the overlay visualization in this substage
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from geometry import GeometryBranchResult
from roi_crop import ROICropResult
from color_branch import TrafficLightCandidate

# =============================================================================
# Debug Artifact Names
# =============================================================================
OVERLAY_SUFFIX = "_overlay.png"
SUMMARY_SUFFIX = "_summary.txt"

# =============================================================================
# Detection Classes
# =============================================================================
# Which ROI each detection class is cut from. Used to attach source_roi and
# source_rect so a detection can be placed in the frame without a lookup
# table on the consumer's side.
ROI_FOR_TYPE = {
    "traffic_light": "traffic",
    "lane_boundary": "lane",
    "stop_sign": "sign",
}

# =============================================================================
# Output Dataclasses
# =============================================================================
@dataclass(frozen=True)
class DetectionObject:
    """
    A single detection in Phase 2 output

    type: detection class --- "traffic_light", "lane_boundary", "stop_sign"
    label_detail: branch-specific label (light color, boundary type, sign type)
    confidence: detection confidence in [0, 1]
    position: {"x", "y"} centroid of bounding_box, ROI-LOCAL
    bounding_box: (x, y, w, h) ROI-local; internal debug field for the overlay
    source_roi: which ROI this came from --- "lane", "traffic", "sign"
    source_rect: (x, y, w, h) of that ROI in frame coordinates. Add the origin
                 to position to get frame coordinates
    frame_id: the frame this detection was made on
    timestamp: the frame's capture timestamp in ms, not a per-branch clock
    """
    type: str
    label_detail: str
    confidence: float
    position: dict
    bounding_box: tuple
    source_roi: str
    source_rect: tuple
    frame_id: int
    timestamp: int

@dataclass(frozen=True)
class FusionResult:
    """
    Output of the feature fusion stage

    detections: every detection that survived conflict resolution, in the
                order traffic_light, lane_boundary (descending confidence),
                stop_sign
    frame_id: carried from ROICropResult, never re-derived
    timestamp_ms: carried from ROICropResult, never re-derived
    """
    detections: List[DetectionObject]
    frame_id: int
    timestamp_ms: int

# =============================================================================
# Utility Functions
# =============================================================================
def _centroid(
        bbox: tuple
    ) -> dict:
    """
    Purpose:
        Compute bounding box centroid. Guards against zero-size bbox.
    """
    x, y, w, h = bbox
    cx = float(x) + float(w) / 2.0 if w > 0 else float(x)
    cy = float(y) + float(h) / 2.0 if h > 0 else float(y)
    return {"x": round(cx, 2), "y": round(cy, 2)}

def _valid_confidence(
        c: float
    ) -> bool:
    """
    Purpose:
        Guard against invalid confidence values
    """
    return 0.0 <= c <= 1.0

def _best_candidate(
        candidates: list,
        log: list,
        class_name: str
    ):
    """
    Purpose:
        Select the best candidate from a list of candidates
    """
    valid = []
    for cand in candidates:
        if not _valid_confidence(cand.confidence):
            log.append(
                f"[DISCARD] {class_name}: invalid confidence {cand.confidence:.4f} "
                f"frame_id={cand.frame_id}"
            )
            continue
        valid.append(cand)

    if not valid:
        return None
    return max(valid, key=lambda c: c.confidence)

def _rects(
        roi: ROICropResult
    ) -> dict:
    """
    Purpose:
        Map ROI name to its (x, y, w, h) in frame coordinates
    """
    return {
        "lane": roi.lane_rect,
        "traffic": roi.traffic_rect,
        "sign": roi.sign_rect,
    }

def _detection(
        det_type: str,
        cand,
        rects: dict,
        frame_id: int,
        timestamp_ms: int,
    ) -> DetectionObject:
    """
    Purpose:
        Build one DetectionObject from a branch candidate, attaching the
        coordinate space it belongs to and the frame's stamp

    Notes:
        The stamp comes from the frame, not from cand.timestamp_ms. The two
        agree today, but taking it from the frame means three detections fused
        from one capture cannot disagree if a branch ever samples its own clock
    """
    roi_name = ROI_FOR_TYPE[det_type]
    return DetectionObject(
        type = det_type,
        label_detail = cand.label,
        confidence = cand.confidence,
        position = _centroid(cand.bbox),
        bounding_box = cand.bbox,
        source_roi = roi_name,
        source_rect = rects[roi_name],
        frame_id = frame_id,
        timestamp = timestamp_ms,
    )

# =============================================================================
# Validation
# =============================================================================
def _validate(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
    ) -> None:
    """
    Purpose:
        Reject inputs that cannot be fused, naming what is wrong

    Notes:
        The stamp check is the one that matters. Both branches descend from the
        same ROICropResult, so a disagreement means candidates from two
        different frames reached one fusion call, which would produce a
        DetectionObject labelled with a frame it did not come from
    """
    if geometry is None:
        raise ValueError("fuse_detections: geometry result is None")
    if roi is None:
        raise ValueError("fuse_detections: roi result is None")
    if (geometry.frame_id, geometry.timestamp_ms) != (roi.frame_id, roi.timestamp_ms):
        raise ValueError(
            f"fuse_detections: geometry stamp "
            f"{(geometry.frame_id, geometry.timestamp_ms)} does not match roi stamp "
            f"{(roi.frame_id, roi.timestamp_ms)} — candidates are from different frames"
        )

# =============================================================================
# Feature Fusion Stage
# =============================================================================
def fuse_detections(
        geometry: GeometryBranchResult,
        traffic_candidates: list,
        roi: ROICropResult,
    ) -> tuple:
    """
    Purpose:
        Fuse candidates from the color and geometry branches into unified
        DetectionObjects

    Inputs:
        geometry: GeometryBranchResult from run_geometry_stage()
        traffic_candidates: list[TrafficLightCandidate] from the color branch.
                            A candidate must expose label, bbox, confidence
                            and frame_id
        roi: ROICropResult, the source of the frame stamp and the ROI rects

    Outputs:
        result: FusionResult
        debug_summary: dict --- "frame_id", "timestamp_ms", "counts", "total",
                       "discarded", "suppressed", "log"

    Conflict resolution:
        traffic_light: highest confidence wins, the rest are suppressed
        stop_sign: highest confidence wins, the rest are suppressed
        lane_boundary: every valid candidate is forwarded, sorted descending
        Any candidate with a confidence outside [0, 1] is discarded and logged
    """
    _validate(geometry, roi)

    log = []
    detections = []

    frame_id = roi.frame_id
    timestamp_ms = roi.timestamp_ms
    rects = _rects(roi)

    lane_candidates = geometry.lane_candidates
    sign_candidates = geometry.sign_candidates

    # ========================================================================
    # Traffic Light Fusion
    # Determine best traffic light candidate, if any, and log discards
    # ========================================================================
    best_tl = _best_candidate(traffic_candidates, log, "traffic_light")

    if best_tl is not None:
        # Log losers
        for c in traffic_candidates:
            if c is not best_tl and _valid_confidence(c.confidence):
                log.append(
                    f"[SUPPRESSED] traffic_light: {c.label} conf={c.confidence:.4f} "
                    f"(lost to {best_tl.label} conf={best_tl.confidence:.4f})"
                )
        detections.append(
            _detection("traffic_light", best_tl, rects, frame_id, timestamp_ms)
        )

    # ========================================================================
    # Lane Boundary Fusion
    # Every valid candidate is forwarded; only invalid ones are discarded
    # ========================================================================
    valid_lanes = [c for c in lane_candidates if _valid_confidence(c.confidence)]
    invalid_lanes = [c for c in lane_candidates if not _valid_confidence(c.confidence)]

    for c in invalid_lanes:
        log.append(
            f"[DISCARD] lane_boundary: invalid confidence {c.confidence:.4f} "
            f"frame_id={c.frame_id}"
        )

    # Sort descending by confidence (LB-2)
    for c in sorted(valid_lanes, key=lambda x: x.confidence, reverse=True):
        detections.append(
            _detection("lane_boundary", c, rects, frame_id, timestamp_ms)
        )

    # ========================================================================
    # Stop Sign Fusion
    # Determine best stop sign candidate, if any, and log discards
    # ========================================================================
    best_sign = _best_candidate(sign_candidates, log, "stop_sign")

    if best_sign is not None:
        for c in sign_candidates:
            if c is not best_sign and _valid_confidence(c.confidence):
                log.append(
                    f"[SUPPRESSED] stop_sign: conf={c.confidence:.4f} v={c.vertex_count} "
                    f"(lost to conf={best_sign.confidence:.4f} v={best_sign.vertex_count})"
                )
        detections.append(
            _detection("stop_sign", best_sign, rects, frame_id, timestamp_ms)
        )

    # ========================================================================
    # Debug Results
    # ========================================================================
    type_counts = {}
    for d in detections:
        type_counts[d.type] = type_counts.get(d.type, 0) + 1

    debug_summary = {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "counts": type_counts,
        "total": len(detections),
        "discarded": sum(1 for entry in log if "[DISCARD]" in entry),
        "suppressed": sum(1 for entry in log if "[SUPPRESSED]" in entry),
        "log": log,
    }

    return FusionResult(
        detections = detections,
        frame_id = frame_id,
        timestamp_ms = timestamp_ms,
    ), debug_summary

# =============================================================================
# Debug Visualization
# =============================================================================
_TYPE_COLORS = {
    "traffic_light": (255,  0,  0),   # blue
    "lane_boundary": (0,  255,  0),   # green
    "stop_sign":     (0,    0, 255),  # red
}

def draw_fusion_overlay(
        canvas: np.ndarray,
        detections: List[DetectionObject],
        title: str = "",
        source_roi: Optional[str] = None,
    ) -> np.ndarray:
    """
    Purpose:
        Draw bounding boxes, type labels, and confidence scores on a copy of
        canvas. Uses bounding_box for debugging purposes

    Inputs:
        canvas: image to draw on, at the resolution of one ROI
        detections: detections to draw
        title: optional caption
        source_roi: draw only detections from this ROI ("lane", "traffic",
                    "sign"). Coordinates are ROI-local, so drawing a sign
                    detection on a lane canvas places the box at a meaningless
                    position. None draws everything, which is only correct when
                    every detection shares one ROI

    Outputs:
        annotated copy; the input is not modified
    """
    vis = canvas.copy()
    for d in detections:
        if source_roi is not None and d.source_roi != source_roi:
            continue
        color = _TYPE_COLORS.get(d.type, (200, 200, 200))
        x, y, w, h = d.bounding_box
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), color, 2)
        label_text = f"{d.type}:{d.label_detail} {d.confidence:.2f}"
        cv2.putText(vis, label_text, (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        # Mark centroid
        cx = int(d.position["x"])
        cy = int(d.position["y"])
        cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 8, 1)
    if title:
        cv2.putText(vis, title, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return vis

# =============================================================================
# Self-test
#
#   python3 feature_fusion.py              logic tests only, runs anywhere
#   python3 feature_fusion.py --artifacts  also writes overlays and summaries
#
# The mock frames cover:
#   f0: all three detection types present
#   f1: conflicting traffic-light colors (TL-1 suppression)
#   f2: multiple stop-sign candidates (SS-1 suppression)
#   f3: multiple lane boundaries (LB-1 all forwarded)
#   f4: no candidates (empty output)
#   f5: invalid confidence values (F2 discard)
#
# Exits non-zero on any logic failure so it can gate a commit.
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    import traceback

    from capture import FrameData
    from preprocess import preprocess_frame
    from roi_crop import crop_rois
    from geometry import LaneCandidate, SignCandidate

    OUTPUT_DIR = "vision_stack/frames/mock/results"

    H, W = 360, 480
    FID, TS = 7, 1000

    _results = []

    def check(name, fn):
        """Run one test, record pass/fail, never abort the suite."""
        try:
            fn()
        except Exception:
            _results.append((name, False))
            print(f"  FAIL  {name}")
            for line in traceback.format_exc().strip().splitlines()[-2:]:
                print(f"        {line.strip()}")
        else:
            _results.append((name, True))
            print(f"  ok    {name}")

    def expect_raises(exc_type, fn):
        try:
            fn()
        except exc_type:
            return
        raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")

    # =========================================================================
    # Mock Object Generators
    # =========================================================================
    def _roi(frame_id=FID, timestamp_ms=TS):
        """An ROICropResult built by running the real stages in order."""
        rng = np.random.default_rng(2)
        bgr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        return crop_rois(preprocess_frame(FrameData(bgr, frame_id, timestamp_ms)))

    def _tl(label, bbox, conf, fid=FID, ts=TS):
        """Mock traffic light candidate."""
        return TrafficLightCandidate(label=label, bbox=bbox, confidence=conf,
                                     frame_id=fid, timestamp_ms=ts)

    def _lane(bbox, conf, fid=FID, ts=TS):
        """Mock lane boundary candidate."""
        return LaneCandidate(label="lane_boundary", bbox=bbox, contour=None,
                             confidence=conf, frame_id=fid, timestamp_ms=ts)

    def _sign(bbox, conf, verts=8, fid=FID, ts=TS):
        """Mock stop sign candidate."""
        return SignCandidate(label="stop_sign", bbox=bbox, contour=None,
                             vertex_count=verts, confidence=conf,
                             frame_id=fid, timestamp_ms=ts)

    def _geo(lanes=(), signs=(), frame_id=FID, timestamp_ms=TS):
        """Mock GeometryBranchResult."""
        return GeometryBranchResult(list(lanes), list(signs), frame_id, timestamp_ms)

    # =========================================================================
    # Mock Frames
    # =========================================================================
    mock_frames = [
        {
            "name": "f0_all_present",
            "traffic": [_tl("green", (20, 10, 30, 30), 0.85)],
            "lanes": [_lane((20, 60, 130, 6), 0.72), _lane((280, 55, 120, 6), 0.68)],
            "signs": [_sign((60, 20, 80, 80), 0.78)],
        },
        {
            "name": "f1_tl_conflict",
            "traffic": [
                _tl("red", (20, 10, 30, 30), 0.91),
                _tl("green", (22, 12, 28, 28), 0.55),   # suppressed by TL-1
            ],
            "lanes": [_lane((40, 70, 110, 5), 0.60)],
            "signs": [],
        },
        {
            "name": "f2_multi_sign",
            "traffic": [],
            "lanes": [],
            "signs": [
                _sign((60, 20, 80, 80), 0.82),
                _sign((55, 18, 85, 85), 0.44),   # suppressed by SS-1
            ],
        },
        {
            "name": "f3_multi_lane",
            "traffic": [_tl("yellow", (20, 10, 30, 30), 0.70)],
            "lanes": [
                _lane((15, 65, 140, 6), 0.88),
                _lane((250, 60, 130, 6), 0.79),
                _lane((180, 80, 40, 5), 0.45),
            ],
            "signs": [],
        },
        {
            "name": "f4_no_candidates",
            "traffic": [],
            "lanes": [],
            "signs": [],
        },
        {
            "name": "f5_invalid_confidence",
            "traffic": [_tl("red", (20, 10, 30, 30), -0.5)],   # F2: discarded
            "lanes": [_lane((15, 65, 140, 6), 1.3)],           # F2: discarded
            "signs": [_sign((60, 20, 80, 80), 0.65)],
        },
    ]

    def _fuse(frame, roi=None):
        roi = roi if roi is not None else _roi()
        return fuse_detections(
            _geo(frame["lanes"], frame["signs"]), frame["traffic"], roi
        )

    def _by_name(name):
        return next(f for f in mock_frames if f["name"] == name)

    # -------------------------------------------------------------------------
    # Conflict resolution
    # -------------------------------------------------------------------------
    def t_all_three_types_survive():
        result, summary = _fuse(_by_name("f0_all_present"))
        assert summary["counts"] == {
            "traffic_light": 1, "lane_boundary": 2, "stop_sign": 1
        }, f"got {summary['counts']}"

    def t_traffic_light_conflict_keeps_highest():
        result, summary = _fuse(_by_name("f1_tl_conflict"))
        tl = [d for d in result.detections if d.type == "traffic_light"]
        assert len(tl) == 1 and tl[0].label_detail == "red", f"got {tl}"
        assert summary["suppressed"] == 1, f"got {summary['suppressed']}"

    def t_multiple_signs_keep_highest():
        result, summary = _fuse(_by_name("f2_multi_sign"))
        signs = [d for d in result.detections if d.type == "stop_sign"]
        assert len(signs) == 1 and signs[0].confidence == 0.82
        assert summary["suppressed"] == 1

    def t_all_lanes_forwarded_in_descending_order():
        result, _ = _fuse(_by_name("f3_multi_lane"))
        confs = [d.confidence for d in result.detections if d.type == "lane_boundary"]
        assert confs == [0.88, 0.79, 0.45], f"got {confs}"

    def t_empty_frame_produces_no_detections():
        result, summary = _fuse(_by_name("f4_no_candidates"))
        assert result.detections == []
        assert summary["total"] == 0 and summary["counts"] == {}

    def t_invalid_confidence_is_discarded():
        result, summary = _fuse(_by_name("f5_invalid_confidence"))
        assert summary["discarded"] == 2, f"got {summary['discarded']}"
        assert summary["counts"] == {"stop_sign": 1}, f"got {summary['counts']}"

    # -------------------------------------------------------------------------
    # Empty frames still carry identity
    # -------------------------------------------------------------------------
    def t_empty_frame_still_carries_the_stamp():
        """A frame with no detections is the one you most want stamped."""
        result, summary = _fuse(_by_name("f4_no_candidates"))
        assert (result.frame_id, result.timestamp_ms) == (FID, TS)
        assert (summary["frame_id"], summary["timestamp_ms"]) == (FID, TS)

    # -------------------------------------------------------------------------
    # Coordinate space
    # -------------------------------------------------------------------------
    def t_every_detection_names_its_roi():
        result, _ = _fuse(_by_name("f0_all_present"))
        expected = {"traffic_light": "traffic", "lane_boundary": "lane",
                    "stop_sign": "sign"}
        for d in result.detections:
            assert d.source_roi == expected[d.type], (
                f"{d.type} claims roi {d.source_roi}"
            )

    def t_source_rect_matches_the_roi_it_names():
        roi = _roi()
        result, _ = _fuse(_by_name("f0_all_present"), roi)
        rects = {"lane": roi.lane_rect, "traffic": roi.traffic_rect,
                 "sign": roi.sign_rect}
        for d in result.detections:
            assert d.source_rect == rects[d.source_roi], (
                f"{d.type} rect {d.source_rect} != {rects[d.source_roi]}"
            )

    def t_frame_conversion_lands_inside_the_frame():
        """The documented conversion must produce a point on the image."""
        result, _ = _fuse(_by_name("f0_all_present"))
        for d in result.detections:
            fx = d.position["x"] + d.source_rect[0]
            fy = d.position["y"] + d.source_rect[1]
            assert 0 <= fx < W and 0 <= fy < H, f"{d.type} maps to ({fx}, {fy})"

    def t_position_is_the_bbox_centroid():
        result, _ = _fuse(_by_name("f2_multi_sign"))
        d = result.detections[0]
        assert d.position == {"x": 100.0, "y": 60.0}, f"got {d.position}"

    # -------------------------------------------------------------------------
    # Stamp handling
    # -------------------------------------------------------------------------
    def t_detections_share_one_stamp():
        """Three detections from one capture must not disagree on when."""
        result, _ = _fuse(_by_name("f0_all_present"))
        stamps = {(d.frame_id, d.timestamp) for d in result.detections}
        assert stamps == {(FID, TS)}, f"detections disagree: {stamps}"

    def t_stamp_comes_from_the_frame_not_the_candidate():
        """A candidate carrying a stale stamp must not relabel the frame."""
        frame = {
            "traffic": [],
            "lanes": [_lane((15, 65, 140, 6), 0.9, fid=99, ts=123456)],
            "signs": [],
        }
        roi = _roi()
        result, _ = fuse_detections(
            _geo(frame["lanes"], frame["signs"], frame_id=FID, timestamp_ms=TS),
            frame["traffic"], roi,
        )
        d = result.detections[0]
        assert (d.frame_id, d.timestamp) == (FID, TS), (
            f"candidate stamp leaked through: {(d.frame_id, d.timestamp)}"
        )

    def t_cross_frame_inputs_rejected():
        """Geometry and ROI from different frames must not silently fuse."""
        expect_raises(ValueError, lambda: fuse_detections(
            _geo(frame_id=4, timestamp_ms=400), [], _roi(frame_id=5, timestamp_ms=500)
        ))

    def t_stamp_survives_the_whole_chain():
        """capture -> preprocess -> crop -> geometry -> fusion, one stamp."""
        from geometry import run_geometry_stage
        rng = np.random.default_rng(4)
        bgr = rng.integers(0, 120, (H, W, 3), dtype=np.uint8)
        bgr[300:305, 60:190] = 235
        fd = FrameData(bgr, 21, 654321)
        roi = crop_rois(preprocess_frame(fd))
        geo, _, _ = run_geometry_stage(roi)
        result, summary = fuse_detections(geo, [], roi)
        assert (result.frame_id, result.timestamp_ms) == (21, 654321)
        assert (summary["frame_id"], summary["timestamp_ms"]) == (21, 654321)

    # -------------------------------------------------------------------------
    # Result contract
    # -------------------------------------------------------------------------
    def t_result_is_immutable():
        result, _ = _fuse(_by_name("f0_all_present"))
        expect_raises(Exception, lambda: setattr(result, "frame_id", 0))

    def t_detection_is_immutable():
        result, _ = _fuse(_by_name("f0_all_present"))
        expect_raises(Exception,
                      lambda: setattr(result.detections[0], "confidence", 0.0))

    def t_none_inputs_rejected():
        expect_raises(ValueError, lambda: fuse_detections(None, [], _roi()))
        expect_raises(ValueError, lambda: fuse_detections(_geo(), [], None))

    # -------------------------------------------------------------------------
    # Overlay
    # -------------------------------------------------------------------------
    def t_overlay_filters_by_roi():
        """Drawing every type on one ROI canvas places boxes at nonsense spots."""
        result, _ = _fuse(_by_name("f0_all_present"))
        canvas = np.zeros((108, 432, 3), np.uint8)
        all_types = draw_fusion_overlay(canvas, result.detections)
        lanes_only = draw_fusion_overlay(canvas, result.detections, source_roi="lane")
        assert not np.array_equal(all_types, lanes_only), "filter had no effect"

    def t_overlay_does_not_modify_input():
        result, _ = _fuse(_by_name("f0_all_present"))
        canvas = np.zeros((108, 432, 3), np.uint8)
        before = canvas.copy()
        draw_fusion_overlay(canvas, result.detections, source_roi="lane")
        assert np.array_equal(canvas, before), "overlay drew into its input"

    # -------------------------------------------------------------------------
    print("\nConflict resolution")
    check("all three types survive",             t_all_three_types_survive)
    check("TL conflict keeps highest",           t_traffic_light_conflict_keeps_highest)
    check("multiple signs keep highest",         t_multiple_signs_keep_highest)
    check("all lanes forwarded, descending",     t_all_lanes_forwarded_in_descending_order)
    check("empty frame yields no detections",    t_empty_frame_produces_no_detections)
    check("invalid confidence discarded",        t_invalid_confidence_is_discarded)

    print("\nCoordinate space")
    check("every detection names its ROI",       t_every_detection_names_its_roi)
    check("source_rect matches that ROI",        t_source_rect_matches_the_roi_it_names)
    check("conversion lands inside the frame",   t_frame_conversion_lands_inside_the_frame)
    check("position is the bbox centroid",       t_position_is_the_bbox_centroid)

    print("\nStamp")
    check("empty frame still carries a stamp",   t_empty_frame_still_carries_the_stamp)
    check("detections share one stamp",          t_detections_share_one_stamp)
    check("stamp comes from the frame",          t_stamp_comes_from_the_frame_not_the_candidate)
    check("cross-frame inputs rejected",         t_cross_frame_inputs_rejected)
    check("stamp survives the whole chain",      t_stamp_survives_the_whole_chain)

    print("\nResult contract")
    check("FusionResult is immutable",           t_result_is_immutable)
    check("DetectionObject is immutable",        t_detection_is_immutable)
    check("None inputs rejected",                t_none_inputs_rejected)

    print("\nOverlay")
    check("filters by source ROI",               t_overlay_filters_by_roi)
    check("does not modify input",               t_overlay_does_not_modify_input)

    passed = sum(1 for _, ok in _results if ok)
    print(f"\n{passed}/{len(_results)} passed")

    # -------------------------------------------------------------------------
    # Artifact pass: write per-frame overlays and summaries
    # -------------------------------------------------------------------------
    if "--artifacts" not in sys.argv:
        print("\nArtifacts: skipped (pass --artifacts to write overlays)")
        sys.exit(0 if passed == len(_results) else 1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    roi = _roi()

    print("\nArtifacts")
    for i, frame in enumerate(mock_frames):
        geo = _geo(frame["lanes"], frame["signs"], frame_id=roi.frame_id,
                   timestamp_ms=roi.timestamp_ms)
        result, summary = fuse_detections(geo, frame["traffic"], roi)

        txt_path = os.path.join(OUTPUT_DIR, f"fusion_{frame['name']}{SUMMARY_SUFFIX}")
        with open(txt_path, "w") as f:
            f.write(f"Frame: {summary['frame_id']}\n")
            f.write(f"Timestamp: {summary['timestamp_ms']}\n")
            f.write(f"Total out: {summary['total']}\n")
            f.write(f"Counts: {summary['counts']}\n")
            f.write(f"Discarded: {summary['discarded']}\n")
            f.write(f"Suppressed: {summary['suppressed']}\n")
            f.write("\nDetections:\n")
            for d in result.detections:
                f.write(
                    f"type={d.type:<16} label={d.label_detail:<12} "
                    f"conf={d.confidence:.4f} roi={d.source_roi:<8} "
                    f"pos={d.position} bbox={d.bounding_box} "
                    f"rect={d.source_rect}\n"
                )
            f.write("\nLog:\n")
            for entry in summary["log"]:
                f.write(f"{entry}\n")

        # One overlay per ROI, since coordinates are ROI-local. Drawing a sign
        # detection on the lane canvas would place it at a meaningless spot.
        canvases = {
            "lane": np.zeros((*roi.lane_roi.shape[:2], 3), np.uint8),
            "traffic": np.zeros((*roi.traffic_roi.shape[:2], 3), np.uint8),
            "sign": np.zeros((*roi.sign_roi.shape[:2], 3), np.uint8),
        }
        for roi_name, canvas in canvases.items():
            vis = draw_fusion_overlay(
                canvas, result.detections,
                title=f"{frame['name']} [{roi_name}]",
                source_roi=roi_name,
            )
            cv2.imwrite(
                os.path.join(
                    OUTPUT_DIR, f"fusion_{frame['name']}_{roi_name}{OVERLAY_SUFFIX}"
                ),
                vis,
            )

        count_str = ", ".join(f"{k}={v}" for k, v in summary["counts"].items()) or "none"
        print(
            f"[OK] {frame['name']} detections=[{count_str}] "
            f"discarded={summary['discarded']} suppressed={summary['suppressed']}"
        )
        for entry in summary["log"]:
            print(f"     {entry}")

    print(f"\nDone. {len(mock_frames)} mock frames written to {OUTPUT_DIR}")
    sys.exit(0 if passed == len(_results) else 1)