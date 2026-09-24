"""Feature fusion: one frame's branch candidates as a single list of detections.

Purpose:
    The geometry and color branches run independently and emit their own
    candidate types in their own ROI spaces. Fusion converts them into the
    single Phase 2 DetectionObject schema, resolves conflicts within each
    class, and logs every discard and suppression. color_branch is
    deliberately not imported, so lane and sign fusion work without it.

Main package:
    FusionResult: the frame's surviving DetectionObjects, ordered traffic
    light, lane boundaries by descending confidence, stop sign, and stamped
    with the ROICropResult's frame identity. Positions are ROI-local; each
    detection carries the rect that maps it into the frame.

Flow:
    1. Check that the geometry result and the ROI crop share a frame stamp.
    2. Traffic light: keep the highest-confidence valid candidate.
    3. Lane boundaries: forward every valid candidate, by descending confidence.
    4. Stop sign: keep the highest-confidence valid candidate.
    5. Package with the frame identity and a debug summary.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from src.params import (
    LANE_BOUNDARY, ROI_LANE, ROI_SIGN, ROI_TRAFFIC, STOP_SIGN, TRAFFIC_LIGHT,
    FUSION_OVERLAY_SUFFIX as OVERLAY_SUFFIX, FUSION_SUMMARY_SUFFIX as SUMMARY_SUFFIX,
)
from src.perception.geometry import GeometryBranchResult
from src.utils import check_same_frame
from src.perception.roi_crop import ROICropResult

# Which ROI each detection class is cut from. Used to attach source_roi and
# source_rect so a detection can be placed in the frame without a lookup
# table on the consumer's side.
ROI_FOR_TYPE = {
    TRAFFIC_LIGHT: ROI_TRAFFIC,
    LANE_BOUNDARY: ROI_LANE,
    STOP_SIGN: ROI_SIGN,
}


@dataclass(frozen=True)
class DetectionObject:
    """
    One Phase 2 detection.

    Coordinates are ROI-local: a lane_boundary at (200, 50) and a stop_sign at
    (200, 50) are not the same place. To get frame coordinates, add the
    source_rect origin:

        frame_x = detection.position["x"] + detection.source_rect[0]
        frame_y = detection.position["y"] + detection.source_rect[1]

    The rect rides on each detection rather than the envelope so a detection
    stays self-describing in navigation, where the ROICropResult is out of scope.
    """
    type: str                                   # "traffic_light" | "lane_boundary" | "stop_sign"
    label_detail: str                           # branch label: light color, boundary type, sign type
    confidence: float                           # [0, 1]
    position: dict[str, float]                  # {"x", "y"}: bounding_box centroid, ROI-local
    bounding_box: tuple[int, int, int, int]     # (x, y, w, h), ROI-local; used only by the debug overlay
    source_roi: str                             # "lane" | "traffic" | "sign"
    source_rect: tuple[int, int, int, int]      # (x, y, w, h) of that ROI in frame px
    frame_id: int
    timestamp: int                              # the frame's capture time in ms, not a branch clock

@dataclass(frozen=True)
class FusionResult:
    """Output of the fusion stage. frame_id and timestamp_ms are copied from ROICropResult, never re-derived."""
    detections: list[DetectionObject]   # traffic_light, then lane_boundary by descending confidence, then stop_sign
    frame_id: int
    timestamp_ms: int


def _centroid(
        bbox: tuple[int, int, int, int]
    ) -> dict[str, float]:
    """{"x", "y"} center of an (x, y, w, h) box, rounded to 0.01. A zero-size side falls back to its origin."""
    x, y, w, h = bbox
    cx = float(x) + float(w) / 2.0 if w > 0 else float(x)
    cy = float(y) + float(h) / 2.0 if h > 0 else float(y)
    return {"x": round(cx, 2), "y": round(cy, 2)}

def _valid_confidence(
        c: float
    ) -> bool:
    """True for a confidence in [0, 1]. NaN fails."""
    return 0.0 <= c <= 1.0

def _best_candidate(
        candidates: list,
        log: list[str],
        class_name: str
    ):
    """Highest-confidence valid candidate (ties keep the first), or None. Appends a discard line to log for each invalid one."""
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
    ) -> dict[str, tuple[int, int, int, int]]:
    """ROI name -> that ROI's (x, y, w, h) in frame px."""
    return {
        ROI_LANE: roi.lane_rect,
        ROI_TRAFFIC: roi.traffic_rect,
        ROI_SIGN: roi.sign_rect,
    }

def _detection(
        det_type: str,
        cand,
        rects: dict[str, tuple[int, int, int, int]],
        frame_id: int,
        timestamp_ms: int,
    ) -> DetectionObject:
    """One DetectionObject from a branch candidate, with its ROI's rect and the frame's stamp."""
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
        # From the frame, not cand.timestamp_ms. The two agree today, but this
        # keeps one capture's detections consistent if a branch ever samples its own clock.
        timestamp = timestamp_ms,
    )


def fuse_detections(
        geometry: GeometryBranchResult,
        traffic_candidates: list,
        roi: ROICropResult,
    ) -> tuple[FusionResult, dict]:
    """
    Fuse one frame's branch candidates into DetectionObjects.

    Purpose:
        Conflict resolution per class. traffic_light and stop_sign: the
        highest confidence wins and the rest are suppressed. lane_boundary:
        every valid candidate is forwarded, sorted descending. In every
        class, a confidence outside [0, 1] is discarded and logged.

    Inputs:
        geometry: From run_geometry_stage().
        traffic_candidates: From run_color_stage(), or [] while the color
            branch is off. Duck-typed: each needs label, bbox, confidence
            and frame_id.
        roi: Supplies the frame identity and the ROI rects.

    Outputs:
        (result, debug_summary). debug_summary holds frame_id, timestamp_ms,
        counts (per type), total, discarded, suppressed and log.

    Raises:
        ValueError: If either input is None, or their frame stamps disagree.
    """
    check_same_frame(geometry, roi, "fuse_detections")

    log = []
    detections = []

    frame_id = roi.frame_id
    timestamp_ms = roi.timestamp_ms
    rects = _rects(roi)

    lane_candidates = geometry.lane_candidates
    sign_candidates = geometry.sign_candidates

    best_tl = _best_candidate(traffic_candidates, log, TRAFFIC_LIGHT)

    if best_tl is not None:
        for c in traffic_candidates:
            if c is not best_tl and _valid_confidence(c.confidence):
                log.append(
                    f"[SUPPRESSED] traffic_light: {c.label} conf={c.confidence:.4f} "
                    f"(lost to {best_tl.label} conf={best_tl.confidence:.4f})"
                )
        detections.append(
            _detection(TRAFFIC_LIGHT, best_tl, rects, frame_id, timestamp_ms)
        )

    valid_lanes = [c for c in lane_candidates if _valid_confidence(c.confidence)]
    invalid_lanes = [c for c in lane_candidates if not _valid_confidence(c.confidence)]

    for c in invalid_lanes:
        log.append(
            f"[DISCARD] lane_boundary: invalid confidence {c.confidence:.4f} "
            f"frame_id={c.frame_id}"
        )

    # LB-2: descending confidence. sorted() is stable, so ties keep input order.
    for c in sorted(valid_lanes, key=lambda x: x.confidence, reverse=True):
        detections.append(
            _detection(LANE_BOUNDARY, c, rects, frame_id, timestamp_ms)
        )

    best_sign = _best_candidate(sign_candidates, log, STOP_SIGN)

    if best_sign is not None:
        for c in sign_candidates:
            if c is not best_sign and _valid_confidence(c.confidence):
                log.append(
                    f"[SUPPRESSED] stop_sign: conf={c.confidence:.4f} v={c.vertex_count} "
                    f"(lost to conf={best_sign.confidence:.4f} v={best_sign.vertex_count})"
                )
        detections.append(
            _detection(STOP_SIGN, best_sign, rects, frame_id, timestamp_ms)
        )

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


_TYPE_COLORS = {
    TRAFFIC_LIGHT: (255,  0,  0),   # blue
    LANE_BOUNDARY: (0,  255,  0),   # green
    STOP_SIGN:     (0,    0, 255),  # red
}

def draw_fusion_overlay(
        canvas: np.ndarray,
        detections: list[DetectionObject],
        title: str = "",
        source_roi: str | None = None,
    ) -> np.ndarray:
    """
    Draw each detection's box, label, confidence and centroid on a copy of canvas.

    Inputs:
        canvas: (h, w, 3) BGR at one ROI's resolution. A gray canvas takes only
            the first BGR component, so most annotations would draw black.
        title: Caption drawn top-left; empty draws none.
        source_roi: Draw only detections from this ROI ("lane", "traffic",
            "sign"). Coordinates are ROI-local, so a sign detection drawn on
            a lane canvas lands somewhere meaningless. None draws everything,
            which is only right when every detection shares one ROI.

    Outputs:
        Annotated copy; the input is untouched.
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
        cx = int(d.position["x"])
        cy = int(d.position["y"])
        cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 8, 1)
    if title:
        cv2.putText(vis, title, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return vis