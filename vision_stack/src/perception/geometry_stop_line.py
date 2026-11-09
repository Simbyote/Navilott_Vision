"""Stop lines from the gray lane ROI: near-horizontal edges paired top with bottom.

Purpose:
    A stop line's edges run across the lane, so their gradient points up or
    down where a lane line's points sideways. Splitting the edges by gradient
    keeps only near-horizontal lines, before any contour can join a stop line
    to the lane lines it touches.

Main package:
    StopLineCandidate: one painted line; clipped when it runs off the view's bottom.
"""
import math

import cv2
import numpy as np
from dataclasses import dataclass

from src.params import STOP_LINE
from src.utils import clamp

from src.perception.geometry_shared import (  # noqa: F401
    StopLineCandidate, StopLineFilter, _contours, _horizontal_edges, _mean_contour_intensity,
)


@dataclass
class _EdgeSegment:
    """One near-horizontal edge, as the line fitted through its points."""
    points: np.ndarray      # (N, 2) float32
    x0: float               # leftmost and rightmost x
    x1: float
    cx: float               # a point on the fitted line
    cy: float
    slope: float            # dy/dx of the fitted line

    def y_at(self, x: float) -> float:
        return self.cy + (x - self.cx) * self.slope

    @property
    def tilt_deg(self) -> float:
        """+ = right end lower, nearer the robot."""
        return math.degrees(math.atan(self.slope))

    @property
    def length_px(self) -> float:
        return (self.x1 - self.x0) / math.cos(math.atan(self.slope))


def _fit_segment(
        points: np.ndarray,
    ) -> _EdgeSegment | None:
    """None if the points span under 2 px in x."""
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    if x1 - x0 < 2.0:
        return None
    vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    if abs(vx) < 1e-6:
        return None
    return _EdgeSegment(points, x0, x1, float(cx), float(cy), float(vy / vx))


def _edge_segments(
        mask: np.ndarray,
        f: StopLineFilter,
    ) -> list[_EdgeSegment]:
    """
    Fitted segments of one polarity, sorted by x0. Pieces broken by a crossing lane line
    (ends within max_thickness_px, lines within 3 px) are refit as one.
    """
    kw, kh = f.close_kernel
    if kw >= 2 or kh >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(kw, 1), max(kh, 1)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    segments = []
    for contour in _contours(mask):
        seg = _fit_segment(contour.reshape(-1, 2).astype(np.float32))
        if seg is not None:
            segments.append(seg)
    segments.sort(key=lambda sg: sg.x0)

    joined = []
    for seg in segments:
        prev = joined[-1] if joined else None
        if (prev is not None
                and 0.0 <= seg.x0 - prev.x1 <= f.max_thickness_px
                and abs(prev.y_at(prev.x1) - seg.y_at(seg.x0)) <= 3.0):
            refit = _fit_segment(np.vstack([prev.points, seg.points]))
            if refit is not None:
                joined[-1] = refit
                continue
        joined.append(seg)
    return joined


def _stop_line_confidence(
        length_px: float,
        tilt_deg: float,
        mean_intensity: float,
        f: StopLineFilter,
    ) -> float:
    """[0, 1]: 50% length, 30% brightness above min_intensity, 20% squareness."""
    length_score = clamp(length_px / max(f.ref_length_px, 1.0), 0.0, 1.0)
    intensity_score = clamp((mean_intensity - f.min_intensity) / max(255.0 - f.min_intensity, 1.0), 0.0, 1.0)
    square_score = clamp(1.0 - abs(tilt_deg) / max(f.max_tilt_deg, 1e-6), 0.0, 1.0)
    return round(clamp(0.5 * length_score + 0.3 * intensity_score + 0.2 * square_score, 0.0, 1.0), 4)


def _stop_line_from(
        top: _EdgeSegment,
        bottoms: list[_EdgeSegment],
        lane_roi: np.ndarray,
        f: StopLineFilter,
        frame_id: int,
        timestamp_ms: int,
    ) -> tuple[StopLineCandidate | None, str]:
    """
    Gate one top edge and pair it with the bottom edge of the same tape: (candidate,
    "accepted") or (None, the failed gate). Shared by both detectors, so their gates can't drift.
    """
    roi_h = lane_roi.shape[0]
    if top.length_px < f.min_length_px:
        return None, "short"
    if abs(top.tilt_deg) > f.max_tilt_deg:
        return None, "tilt"

    best = None
    for b in bottoms:
        if b.length_px < f.min_length_px or abs(b.tilt_deg) > f.max_tilt_deg:
            continue
        x_left, x_right = max(top.x0, b.x0), min(top.x1, b.x1)
        if x_right - x_left < f.min_edge_overlap * min(top.x1 - top.x0, b.x1 - b.x0):
            continue
        mid = (x_left + x_right) / 2.0
        tilt = (top.tilt_deg + b.tilt_deg) / 2.0
        thickness = (b.y_at(mid) - top.y_at(mid)) * math.cos(math.radians(tilt))
        if f.min_thickness_px <= thickness <= f.max_thickness_px \
                and (best is None or thickness < best[1]):
            best = (b, thickness, x_left, x_right, tilt)

    if best is not None:
        b, thickness, x_left, x_right, tilt = best
        y_bottom_at = b.y_at
        clipped = False
    else:
        # The robot may be on the line, its bottom edge below the ROI
        x_left, x_right, tilt = top.x0, top.x1, top.tilt_deg
        mid = (x_left + x_right) / 2.0
        visible = (roi_h - top.y_at(mid)) * math.cos(math.radians(tilt))
        if not (f.min_thickness_px <= visible <= f.max_thickness_px):
            return None, "unpaired"
        thickness = visible
        y_bottom_at = lambda x: float(roi_h)
        clipped = True

    mid = (x_left + x_right) / 2.0
    band = np.array([[x_left, top.y_at(x_left)], [x_right, top.y_at(x_right)],
                     [x_right, y_bottom_at(x_right)], [x_left, y_bottom_at(x_left)]])
    polygon = np.round(band).astype(np.int32).reshape(-1, 1, 2)
    mean_intensity = _mean_contour_intensity(lane_roi, polygon)
    if mean_intensity < f.min_intensity:
        return None, "intensity"

    length_px = (x_right - x_left) / math.cos(math.radians(tilt))
    y_top = min(top.y_at(x_left), top.y_at(x_right))
    y_near = max(y_bottom_at(x_left), y_bottom_at(x_right))
    x, y = int(math.floor(x_left)), int(math.floor(max(y_top, 0.0)))
    w = int(math.ceil(x_right)) - x + 1
    h = int(math.ceil(min(y_near, roi_h))) - y

    return StopLineCandidate(
        label = STOP_LINE,
        bbox = (x, y, w, max(h, 1)),
        x_left = round(x_left, 2),
        x_right = round(x_right, 2),
        y_top_px = round(top.y_at(mid), 2),
        y_bottom_px = round(y_bottom_at(mid), 2),
        y_near_px = round(min(y_near, float(roi_h)), 2),
        tilt_deg = round(tilt, 2),
        length_px = round(length_px, 2),
        thickness_px = round(thickness, 2),
        mean_intensity = round(mean_intensity, 2),
        clipped = clipped,
        confidence = _stop_line_confidence(length_px, tilt, mean_intensity, f),
        frame_id = frame_id,
        timestamp_ms = timestamp_ms,
        proximity = round(clamp(min(y_near, float(roi_h)) / max(roi_h, 1), 0.0, 1.0), 4),
    ), "accepted"


def extract_stop_line_candidates(
        lane_roi: np.ndarray,
        edges_raw: np.ndarray,
        stop_filter: StopLineFilter,
        frame_id: int,
        timestamp_ms: int,
        split: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[list[StopLineCandidate], dict]:
    """(candidates nearest first, debug with the edge split, counts and per-edge trace)."""
    edges_top, edges_bottom = split or _horizontal_edges(lane_roi, edges_raw, stop_filter.max_tilt_deg)
    tops = _edge_segments(edges_top, stop_filter)
    bottoms = _edge_segments(edges_bottom, stop_filter)

    reject_counts = {"seen": len(tops), "short": 0, "tilt": 0, "unpaired": 0,
                     "intensity": 0, "accepted": 0}
    candidates, trace = [], []
    ends = lambda sg: ((sg.x0, sg.y_at(sg.x0)), (sg.x1, sg.y_at(sg.x1)))
    for top in tops:
        candidate, reason = _stop_line_from(top, bottoms, lane_roi, stop_filter,
                                            frame_id, timestamp_ms)
        reject_counts[reason] += 1
        trace.append({"ends": ends(top), "gate": None if candidate else reason,
                      "length": round(top.length_px, 1), "tilt": round(top.tilt_deg, 1),
                      "candidate": candidate})
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda c: -c.y_near_px)

    return candidates, {
        "edges_top": edges_top,
        "edges_bottom": edges_bottom,
        "top_count": len(tops),
        "bottom_count": len(bottoms),
        "reject_counts": reject_counts,
        "trace": trace,
        "bottoms": [ends(b) for b in bottoms],
    }


def find_stop_line_candidates(
        lane_roi: np.ndarray,
        edges_raw: np.ndarray,
        stop_filter: StopLineFilter,
        frame_id: int,
        timestamp_ms: int,
        split: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[StopLineCandidate]:
    """Production twin of extract_stop_line_candidates(), without the debug dict."""
    edges_top, edges_bottom = split or _horizontal_edges(lane_roi, edges_raw, stop_filter.max_tilt_deg)
    bottoms = _edge_segments(edges_bottom, stop_filter)
    candidates = []
    for top in _edge_segments(edges_top, stop_filter):
        candidate, _ = _stop_line_from(top, bottoms, lane_roi, stop_filter,
                                       frame_id, timestamp_ms)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda c: -c.y_near_px)
    return candidates
