"""
feature_fusion.py

Feature Fusion Stage


Purpose:
    The color branch and geometry branch run independently and produce candidates
    in their own ROI coordinate spaces. Fusion has three responsibilities:

    1. Normalize representation — convert all branch-specific candidate types
        into the single Detection Object schema defined by Phase 2. This
        decouples Phase 3 from knowing anything about branch internals.

    2. Per-class conflict resolution — each detection class (traffic_light,
        lane_boundary, stop_sign) may produce multiple candidates in a single
        frame. Fusion selects or aggregates within each class according to
        explicit rules. Candidates from different classes are never merged.

    3. Assign position where available — lane_boundary candidates carry
        bounding box centroids as (x, y) in ROI pixel coords. traffic_light
        and stop_sign likewise. Position is recorded but is still in ROI-
        relative coordinates at this stage; perspective transformation
        (Step 6) converts lane positions to metric coordinates later.

Nothing is discarded silently. Any candidate that does not survive conflict
resolution is logged in the debug summary. Temporal filtering, navigation
decisions, and perspective transformation are all downstream — not here.

Documented Phase 2 output contract (from pipeline.md)
------------------------------------------------------
{
  "type":       "traffic_light | stop_sign | lane_boundary",
  "position":   { "x": float, "y": float },
  "confidence": float,
  "timestamp":  int
}

position is the centroid of the winning candidate's bounding box in ROI
pixel coordinates (integer values stored as float). It is NOT yet in metric
world coordinates — that conversion is the role of Step 6 (perspective
transformation). At this stage, position is a valid intermediate output
field, not debug-only data.

bounding_box is retained in the DetectionObject as an internal debug field.
It is not part of the documented Phase 2 output contract and must not be
forwarded to Phase 3 as a navigation input. It is used only for the overlay
visualisation in this substage.

Input contract
--------------
  traffic_candidates : list[TrafficLightCandidate]
      From color_branch.extract_traffic_light_candidates().
      Fields used: label, bbox, confidence, frame_id, timestamp_ms
      May be empty.

  lane_candidates : list[LaneCandidate]
      From geometry_branch.extract_lane_candidates().
      Fields used: label, bbox, confidence, frame_id, timestamp_ms
      May be empty.

  sign_candidates : list[SignCandidate]
      From geometry_branch.extract_sign_candidates().
      Fields used: label, bbox, confidence, vertex_count, frame_id, timestamp_ms
      May be empty.

  frame_id : int
      From capture loop. Used if candidate lists are empty (no frame_id
      available from candidates directly).

  timestamp_ms : int
      From capture loop. Same fallback purpose.

  source_rois : SourceROIInfo  (optional)
      Holds the shape of each ROI so that bounding-box centroids can be
      computed correctly. If None, position fields are set to (0.0, 0.0)
      and a warning is emitted. This is a debug degradation, not a crash.

Output contract
---------------
  list[DetectionObject]

  Each DetectionObject:
      .type         str    — "traffic_light" | "stop_sign" | "lane_boundary"
      .position     dict   — {"x": float, "y": float} — centroid in ROI px coords
      .confidence   float  — [0.0, 1.0]
      .timestamp    int    — timestamp_ms

      .bounding_box tuple  — (x, y, w, h) INTERNAL DEBUG ONLY.
                             Not forwarded to Phase 3. Used for overlay only.
      .label_detail str    — sub-label from candidate ("red"|"yellow"|"green"
                             for traffic_light; "lane_boundary"; "stop_sign")
                             INTERNAL DEBUG ONLY.

  The list contains at most one DetectionObject per class per frame:
      - at most 1 traffic_light  (highest confidence among red/yellow/green)
      - at most N lane_boundary  (all accepted lane candidates are kept;
                                  no within-class suppression for lane lines
                                  since multiple boundaries are valid)
      - at most 1 stop_sign      (highest confidence)

  Empty list is a valid output (no detections this frame).

Fusion logic rules
------------------
  TRAFFIC LIGHT
    Rule TL-1: If candidates of multiple colors survive blob filtering in the
               same frame, retain only the highest-confidence candidate.
               Rationale: only one light can be active. Multiple surviving
               color blobs indicate a false positive in at least one mask.
    Rule TL-2: If confidence < 0.0 (malformed input), discard silently and
               log in summary.

  LANE BOUNDARY
    Rule LB-1: All lane candidates that survived the geometry branch filter
               are forwarded. No within-class suppression. Multiple parallel
               lane lines are a valid detection state.
    Rule LB-2: Sort by confidence descending before emission so Phase 3
               receives them in priority order.

  STOP SIGN
    Rule SS-1: If multiple sign candidates exist, retain only the
               highest-confidence candidate.
               Rationale: only one stop sign is expected at course exit.
    Rule SS-2: If confidence < 0.0, discard and log.

Labelling rules
---------------
  - type field maps directly from candidate class:
      TrafficLightCandidate  → "traffic_light"
      LaneCandidate          → "lane_boundary"
      SignCandidate          → "stop_sign"
  - label_detail is the original .label field from the candidate (kept
    internally for debug; Phase 3 consumes .type only).
  - position is computed as bounding box centroid:
      x = bbox_x + bbox_w / 2.0
      y = bbox_y + bbox_h / 2.0
  - timestamp is taken from the winning candidate's .timestamp_ms field.
    If all candidate lists are empty, falls back to the timestamp_ms argument.

Failure cases
-------------
  F1 — All three candidate lists are empty:
       Returns empty list. Valid output. Phase 3 handles no-detection frames.

  F2 — Candidate with negative or >1.0 confidence:
       Discarded with a log entry. Does not raise; pipeline continues.

  F3 — source_rois is None:
       position fields default to (0.0, 0.0). Warning printed. Not a crash.
       The overlay visualisation will still draw bounding boxes correctly
       (bbox is taken directly from the candidate).

  F4 — frame_id mismatch across candidate lists:
       Candidates from different lists may carry different frame_ids if the
       pipeline branched asynchronously. fusion uses the frame_id argument
       as the authoritative frame identifier for the output; candidate
       frame_ids are not cross-checked here. That is Phase 3's concern.

  F5 — Candidate bbox contains zero-width or zero-height:
       Centroid computation is guarded; position falls back to (bbox_x, bbox_y).

  F6 — More than one candidate per class with identical confidence:
       The first candidate in the sorted order is selected. No tiebreaker
       beyond sort stability is defined at this stage.
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional

# =============================================================================
# Output Dataclasses
# =============================================================================

@dataclass
class SourceROIInfo:
    """
    Holds ROI shapes needed to contextualize position values

    lane_shape: shape of the lane ROI (H, W)
    traffic_shape: shape of the traffic ROI (H, W)
    sign_shape: shape of the sign ROI (H, W)
    """
    lane_shape: tuple
    traffic_shape: tuple
    sign_shape: tuple


@dataclass
class DetectionObject:
    """
    A single detection in Phase 2 output

    type: candidate type accepted by the detection system
    position: xy coordinates of the centroid of the detection's bounding box
    confidence: detection confidence
    timestamp: time at which the detection was made
    bounding_box: (x, y, w, h) of the detection's bounding box
    label_detail: original .label field from the candidate
    """
    type: str
    position: dict
    confidence: float
    timestamp: int
    bounding_box: tuple
    label_detail: str

@dataclass
class _TrafficLightCandidate:
    """
    Traffic light candidate

    label: traffic light color
    bbox: (x, y, w, h) of the candidate's bounding box
    confidence: detection confidence
    frame_id: frame identifier
    timestamp_ms: time at which the detection was made
    """
    label: str
    bbox: tuple
    confidence: float
    frame_id: int
    timestamp_ms: int

@dataclass
class _LaneCandidate:
    """
    Lane boundary candidate
    
    label: lane boundary type
    bbox: (x, y, w, h) of the candidate's bounding box
    contour: lane boundary contour
    confidence: detection confidence
    frame_id: frame identifier
    timestamp_ms: time at which the detection was made
    """
    label: str
    bbox: tuple
    contour: object
    confidence: float
    frame_id: int
    timestamp_ms: int

@dataclass
class _SignCandidate:
    """
    Stop sign candidate

    label: stop sign type
    bbox: (x, y, w, h) of the candidate's bounding box
    contour: stop sign contour
    vertex_count: number of vertices in the contour
    confidence: detection confidence
    frame_id: frame identifier
    timestamp_ms: time at which the detection was made
    """
    label: str
    bbox: tuple
    contour: object
    vertex_count: int
    confidence: float
    frame_id: int
    timestamp_ms: int

# =============================================================================
# Utility functions
# ============================================================================

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


# =============================================================================
# Feature Fusion
# ============================================================================

def fuse_detections(
    traffic_candidates: list,
    lane_candidates:    list,
    sign_candidates:    list,
    frame_id:           int = 0,
    timestamp_ms:       int = 0,
    source_rois:        Optional[SourceROIInfo] = None,
) -> tuple:
    """
    Purpose:
        Fuse candidates from color and geometry branches into unified DetectionObjects

    Inputs:
        traffic_candidates : list[TrafficLightCandidate]
        lane_candidates    : list[LaneCandidate]
        sign_candidates    : list[SignCandidate]
        frame_id           : authoritative frame identifier for this fusion call
        timestamp_ms       : authoritative timestamp for this fusion call
        source_rois        : optional SourceROIInfo for context; None is accepted

    Outputs:
        detections    : list[DetectionObject]
        debug_summary : dict — "frame_id", "timestamp_ms", "counts", "discarded", "log"
    """
    log        = []
    detections = []

    if source_rois is None:
        log.append(
            "[WARNING] source_rois not provided — position fields will be "
            "(0.0, 0.0) for all detections."
        )

    # ========================================================================
    # Traffic Light  (Rule TL-1, TL-2)
    # Determine best traffic light candidate, if any, and log discards
    # =======================================================================
    best_tl = _best_candidate(traffic_candidates, log, "traffic_light")

    if best_tl is not None:
        # Log losers
        for c in traffic_candidates:
            if c is not best_tl and _valid_confidence(c.confidence):
                log.append(
                    f"[SUPPRESSED] traffic_light: {c.label} conf={c.confidence:.4f} "
                    f"(lost to {best_tl.label} conf={best_tl.confidence:.4f})"
                )
        pos = _centroid(best_tl.bbox) if source_rois is not None else {"x": 0.0, "y": 0.0}
        detections.append(DetectionObject(
            type         = "traffic_light",
            position     = pos,
            confidence   = best_tl.confidence,
            timestamp    = best_tl.timestamp_ms,
            bounding_box = best_tl.bbox,
            label_detail = best_tl.label,
        ))

    # ========================================================================
    # Lane Boundary  (Rule LB-1, LB-2)
    # Determine best lane boundary candidate, if any, and log discards
    # =======================================================================
    valid_lanes = [c for c in lane_candidates if _valid_confidence(c.confidence)]
    invalid_lanes = [c for c in lane_candidates if not _valid_confidence(c.confidence)]

    for c in invalid_lanes:
        log.append(
            f"[DISCARD] lane_boundary: invalid confidence {c.confidence:.4f} "
            f"frame_id={c.frame_id}"
        )

    # Sort descending by confidence (LB-2)
    for c in sorted(valid_lanes, key=lambda x: x.confidence, reverse=True):
        pos = _centroid(c.bbox) if source_rois is not None else {"x": 0.0, "y": 0.0}
        detections.append(DetectionObject(
            type         = "lane_boundary",
            position     = pos,
            confidence   = c.confidence,
            timestamp    = c.timestamp_ms,
            bounding_box = c.bbox,
            label_detail = c.label,
        ))

    # ========================================================================
    # Stop Sign  (Rule SS-1, SS-2)
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
        pos = _centroid(best_sign.bbox) if source_rois is not None else {"x": 0.0, "y": 0.0}
        detections.append(DetectionObject(
            type         = "stop_sign",
            position     = pos,
            confidence   = best_sign.confidence,
            timestamp    = best_sign.timestamp_ms,
            bounding_box = best_sign.bbox,
            label_detail = best_sign.label,
        ))

    # ========================================================================
    # Debug Results
    # ========================================================================
    type_counts = {}
    for d in detections:
        type_counts[d.type] = type_counts.get(d.type, 0) + 1

    debug_summary = {
        "frame_id":     frame_id,
        "timestamp_ms": timestamp_ms,
        "counts":       type_counts,
        "total":        len(detections),
        "discarded":    sum(1 for entry in log if "[DISCARD]" in entry),
        "suppressed":   sum(1 for entry in log if "[SUPPRESSED]" in entry),
        "log":          log,
    }

    return detections, debug_summary


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
) -> np.ndarray:
    """
    Purpose:
        Draw bounding boxes, type labels, and confidence scores on a copy of canvas
        Uses bounding_box for debugging purposes
    """
    vis = canvas.copy()
    for d in detections:
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
# Testing
# ============================================================================

if __name__ == "__main__":
    """
    Standalone Test

    Mocks an expected result from frames
      - Frame with all three detection types present
      - Frame with conflicting traffic-light colors (TL-1 suppressions)
      - Frame with multiple stop-sign candidates (SS-1 suppressions)
      - Frame with multiple lane boundaries (LB-1 all forwarded)
      - Frame with no candidates (empty output)
      - Frame with an invalid confidence value (F2 discard)

    Each mock frame produces:
      - *_roi_traffic.png
      - *_roi_lane.png
      - *_roi_sign.png

    Output directory: 
        vision_stack/frames/mock/results/

    Note:
        When real ROI images are available in trackT3/results/, the harness loads
        the first available *_roi_traffic.png, *_roi_lane.png, *_roi_sign.png
        as canvases for the overlay.
        Otherwise, it falls back to a blank canvas
    """
    import os

    OUTPUT_DIR = "vision_stack/frames/mock/results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =============================================================================
    # Mock candidate factory helpers
    # =============================================================================
    def _tl(label, bbox, conf, fid=0, ts=1000):
        """
        Mock traffic light candidate
        """
        return _TrafficLightCandidate(label=label, bbox=bbox, confidence=conf,
                                      frame_id=fid, timestamp_ms=ts)

    def _lane(bbox, conf, fid=0, ts=1000):
        """
        Mock lane boundary candidate
        """
        return _LaneCandidate(label="lane_boundary", bbox=bbox, contour=None,
                               confidence=conf, frame_id=fid, timestamp_ms=ts)

    def _sign(bbox, conf, verts=8, fid=0, ts=1000):
        """
        Mock stop sign candidate
        """
        return _SignCandidate(label="stop_sign", bbox=bbox, contour=None,
                               vertex_count=verts, confidence=conf,
                               frame_id=fid, timestamp_ms=ts)

    # =============================================================================
    # Mock Frames
    # =============================================================================
    mock_frames = [
        {
            "name":     "f0_all_present",
            "traffic":  [_tl("green", (20, 10, 30, 30), 0.85)],
            "lanes":    [_lane((0, 50, 15, 120), 0.72), _lane((180, 50, 15, 120), 0.68)],
            "signs":    [_sign((60, 20, 80, 80), 0.78)],
        },
        {
            "name":     "f1_tl_conflict",
            "traffic":  [
                _tl("red",   (20, 10, 30, 30), 0.91),
                _tl("green", (22, 12, 28, 28), 0.55),   # suppressed by TL-1
            ],
            "lanes":    [_lane((5, 50, 12, 100), 0.60)],
            "signs":    [],
        },
        {
            "name":     "f2_multi_sign",
            "traffic":  [],
            "lanes":    [],
            "signs":    [
                _sign((60, 20, 80, 80), 0.82),
                _sign((55, 18, 85, 85), 0.44),   # suppressed by SS-1
            ],
        },
        {
            "name":     "f3_multi_lane",
            "traffic":  [_tl("yellow", (20, 10, 30, 30), 0.70)],
            "lanes":    [
                _lane((0,   50, 14, 110), 0.88),
                _lane((186, 50, 14, 110), 0.79),
                _lane((90,  60, 20,  40), 0.45),
            ],
            "signs":    [],
        },
        {
            "name":     "f4_no_candidates",
            "traffic":  [],
            "lanes":    [],
            "signs":    [],
        },
        {
            "name":     "f5_invalid_confidence",
            "traffic":  [_tl("red", (20, 10, 30, 30), -0.5)],   # F2: discarded
            "lanes":    [_lane((0, 50, 14, 110), 1.3)],          # F2: discarded
            "signs":    [_sign((60, 20, 80, 80), 0.65)],
        },
    ]

    # =============================================================================
    # Try to find a real ROI image for canvas; fall back to blank
    # =============================================================================
    def _load_canvas(results_dir, suffix, fallback_shape=(240, 320, 3)):
        if os.path.isdir(results_dir):
            for f in sorted(os.listdir(results_dir)):
                if f.endswith(suffix):
                    img = cv2.imread(os.path.join(results_dir, f))
                    if img is not None:
                        return img
        return np.zeros(fallback_shape, dtype=np.uint8)

    canvas = _load_canvas(OUTPUT_DIR, "_roi_lane.png")

    # =============================================================================
    # Run Mock Frames
    # =============================================================================
    for i, frame in enumerate(mock_frames):
        ts_ms = int(time.time() * 1000) + i * 33

        detections, summary = fuse_detections(
            traffic_candidates = frame["traffic"],
            lane_candidates    = frame["lanes"],
            sign_candidates    = frame["signs"],
            frame_id           = i,
            timestamp_ms       = ts_ms,
            source_rois        = SourceROIInfo(
                lane_shape    = (canvas.shape[0], canvas.shape[1]),
                traffic_shape = (canvas.shape[0] // 2, canvas.shape[1]),
                sign_shape    = (canvas.shape[0], canvas.shape[1] // 2),
            ),
        )

        # Debug summary
        txt_path = os.path.join(OUTPUT_DIR, f"fusion_{frame['name']}_summary.txt")
        with open(txt_path, "w") as f:
            f.write(f"Frame:      {summary['frame_id']}\n")
            f.write(f"Timestamp:  {summary['timestamp_ms']}\n")
            f.write(f"Total out:  {summary['total']}\n")
            f.write(f"Counts:     {summary['counts']}\n")
            f.write(f"Discarded:  {summary['discarded']}\n")
            f.write(f"Suppressed: {summary['suppressed']}\n")
            f.write("\nDetections:\n")
            for d in detections:
                f.write(
                    f"  type={d.type:<16} label={d.label_detail:<12} "
                    f"conf={d.confidence:.4f}  pos={d.position}  "
                    f"bbox={d.bounding_box}\n"
                )
            f.write("\nLog:\n")
            for entry in summary["log"]:
                f.write(f"  {entry}\n")

        # Overlay visualization
        vis = draw_fusion_overlay(canvas, detections, title=frame["name"])
        img_path = os.path.join(OUTPUT_DIR, f"fusion_{frame['name']}_overlay.png")
        cv2.imwrite(img_path, vis)

        count_str = ", ".join(f"{k}={v}" for k, v in summary["counts"].items()) or "none"
        print(
            f"[OK] {frame['name']}  detections=[{count_str}]  "
            f"discarded={summary['discarded']}  suppressed={summary['suppressed']}"
        )
        if summary["log"]:
            for entry in summary["log"]:
                print(f"      {entry}")

    print(f"\nOutputs written to {OUTPUT_DIR}")