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

from src.perception.geometry import GeometryBranchResult
from src.perception.roi_crop import ROICropResult

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
        traffic_candidates: list of TrafficLightCandidate from the color branch,
                            or [] while it is not wired. Duck-typed: a
                            candidate must expose label, bbox, confidence and
                            frame_id. color_branch is deliberately not
                            imported here, so this stage loads and fuses lane
                            and stop-sign detections without it
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