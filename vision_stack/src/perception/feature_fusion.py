"""Feature fusion: one frame's branch candidates as a single list of detections.

Purpose:
    The branches emit their own candidate types in their own ROIs. Fusion
    converts them to the one detection schema, keeps the best traffic light
    and stop sign, forwards every lane boundary, and logs every discard.

Main package:
    FusionResult: the frame's DetectionObjects, positions ROI-local.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from src.params import LANE_BOUNDARY, ROI_LANE, ROI_SIGN, ROI_TRAFFIC, STOP_SIGN, TRAFFIC_LIGHT
from src.perception.geometry import GeometryBranchResult
from src.utils import check_same_frame
from src.perception.roi_crop import ROICropResult

ROI_FOR_TYPE = {
    TRAFFIC_LIGHT: ROI_TRAFFIC,
    LANE_BOUNDARY: ROI_LANE,
    STOP_SIGN: ROI_SIGN,
}


@dataclass(frozen=True)
class DetectionObject:
    """
    Positions are ROI-local; frame x = position["x"] + source_rect[0], and the same for y.
    The rect rides on each detection so it stays placeable in navigation.
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
    detections: list[DetectionObject]   # traffic_light, then lane_boundary by descending confidence, then stop_sign
    frame_id: int
    timestamp_ms: int


def _centroid(
        bbox: tuple[int, int, int, int]
    ) -> dict[str, float]:
    x, y, w, h = bbox
    cx = float(x) + float(w) / 2.0 if w > 0 else float(x)
    cy = float(y) + float(h) / 2.0 if h > 0 else float(y)
    return {"x": round(cx, 2), "y": round(cy, 2)}


def _valid_confidence(
        c: float
    ) -> bool:
    """In [0, 1]; NaN fails."""
    return 0.0 <= c <= 1.0


def _best_candidate(
        candidates: list,
        log: list[str],
        class_name: str
    ):
    """Highest valid confidence, ties keep the first; logs each invalid one."""
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


def _pick_best(
        candidates: list,
    ):
    """Production twin of _best_candidate(), without the log."""
    valid = [cand for cand in candidates if _valid_confidence(cand.confidence)]

    if not valid:
        return None
    return max(valid, key=lambda c: c.confidence)


def _rects(
        roi: ROICropResult
    ) -> dict[str, tuple[int, int, int, int]]:
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
        # The frame's, not the candidate's, so one capture's detections always agree
        timestamp = timestamp_ms,
    )


def fuse_detections(
        geometry: GeometryBranchResult,
        traffic_candidates: list,
        roi: ROICropResult,
    ) -> tuple[FusionResult, dict]:
    """
    (result, debug summary). Traffic light and stop sign: the most confident wins. Lane
    boundaries: all valid ones, by descending confidence. A confidence outside [0, 1] is
    discarded. ValueError if an input is None or the frames disagree.
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

    # sorted() is stable: ties keep input order
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


def fuse(
        geometry: GeometryBranchResult,
        traffic_candidates: list,
        roi: ROICropResult,
    ) -> FusionResult:
    """Production twin of fuse_detections(), without the log."""
    check_same_frame(geometry, roi, "fuse")

    detections = []

    frame_id = roi.frame_id
    timestamp_ms = roi.timestamp_ms
    rects = _rects(roi)

    best_tl = _pick_best(traffic_candidates)
    if best_tl is not None:
        detections.append(
            _detection(TRAFFIC_LIGHT, best_tl, rects, frame_id, timestamp_ms)
        )

    valid_lanes = [c for c in geometry.lane_candidates if _valid_confidence(c.confidence)]

    # sorted() is stable: ties keep input order
    for c in sorted(valid_lanes, key=lambda x: x.confidence, reverse=True):
        detections.append(
            _detection(LANE_BOUNDARY, c, rects, frame_id, timestamp_ms)
        )

    best_sign = _pick_best(geometry.sign_candidates)
    if best_sign is not None:
        detections.append(
            _detection(STOP_SIGN, best_sign, rects, frame_id, timestamp_ms)
        )

    return FusionResult(
        detections = detections,
        frame_id = frame_id,
        timestamp_ms = timestamp_ms,
    )

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
    Each detection on a copy of a BGR canvas at one ROI's resolution. Give source_roi
    unless every detection shares one ROI: positions are ROI-local.
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