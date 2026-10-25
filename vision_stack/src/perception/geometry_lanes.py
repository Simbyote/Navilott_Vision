"""Lane boundaries from the gray lane ROI: Canny edges, gated contours, merged fragments.

Purpose:
    White tape on a dark mat gives strong edges; a contour is kept when its
    area, elongation, span and brightness match tape. Lines across the lane
    come out of this detector's copy of the edges first, so a stop line
    can't close into one contour with the lane lines.

Main package:
    LaneCandidate: one marking, with foot_x where it meets the view's bottom.
"""

import cv2
import numpy as np
from collections.abc import Sequence

from src.params import LANE_BOUNDARY
from src.utils import clamp

from src.perception.geometry_shared import (  # noqa: F401
    CannyParams, LaneCandidate, LaneContourFilter, _canny, _close_edges, _contours, _horizontal_edges,
    _mean_contour_intensity, contour_foot_x,
)


def _extreme_points(
        contour: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    pts = contour.reshape(-1, 2)
    return pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()]


def _merge_collinear(
        candidates: list[LaneCandidate],
        lane_filter: LaneContourFilter,
        roi_shape: tuple[int, int],
        max_gap_px: float = 40.0,
    ) -> list[LaneCandidate]:
    """
    Join fragments of one horizontal line, so lane_offset can't take two pieces of
    it as opposite boundaries. Vertical candidates pass through.
    """
    roi_h, roi_w = roi_shape
    horiz = [c for c in candidates if c.bbox[2] >= c.bbox[3]]
    other = [c for c in candidates if c.bbox[2] <  c.bbox[3]]

    groups = []
    for c in sorted(horiz, key=lambda c: c.bbox[0]):
        c_left, _ = _extreme_points(c.contour)
        for g in groups:
            _, g_right = _extreme_points(g[-1].contour)
            if float(np.hypot(*(c_left - g_right))) <= max_gap_px:
                g.append(c)
                break
        else:
            groups.append([c])

    merged = []
    for g in groups:
        if len(g) == 1:
            merged.append(g[0])
            continue

        pts = np.vstack([c.contour.reshape(-1, 2) for c in g])
        x, y, w, h = cv2.boundingRect(pts)
        _, (rect_w, rect_h), _ = cv2.minAreaRect(pts)
        long_side = max(rect_w, rect_h)
        raw_short = min(rect_w, rect_h)

        # Refilling the joined points would draw a polygon that isn't the line
        total_len = sum(c.length_px for c in g) or 1.0
        mean_intensity = sum(
            c.mean_intensity * c.length_px for c in g
        ) / total_len

        merged.append(LaneCandidate(
            label = LANE_BOUNDARY,
            bbox = (x, y, w, h),
            contour = pts.reshape(-1, 1, 2),
            confidence = _lane_confidence(
                long_side,
                raw_short,
                w >= h,
                mean_intensity,
                roi_h, roi_w,
                lane_filter),
            length_px = round(long_side, 2),
            width_px = round(raw_short, 2),
            frame_id = g[0].frame_id,
            timestamp_ms = g[0].timestamp_ms,
            proximity = round(clamp((y + h) / max(roi_h, 1), 0.0, 1.0), 4),
            mean_intensity = mean_intensity,
            foot_x = contour_foot_x(pts.reshape(-1, 1, 2)),
        ))

    return merged + other


def _lane_confidence(
        long: float,
        short: float,
        horizontal: bool,
        mean_intensity: float,
        roi_h: float,
        roi_w: float,
        f: LaneContourFilter
    ) -> float:
    """[0, 1]: 50% length, 30% brightness above min_intensity, 20% thickness."""
    extent = roi_w if horizontal else roi_h
    ref_len = max(f.ref_length * max(extent, 1), 1.0)
    length_score = clamp(long / ref_len, 0.0, 1.0)

    width_score = clamp(short / max(f.ref_width, 1.0), 0.0, 1.0)

    denom = max(255.0 - f.min_intensity, 1.0)
    intensity_score = clamp((mean_intensity - f.min_intensity) / denom, 0.0, 1.0)

    score = 0.5 * length_score + 0.3 * intensity_score + 0.2 * width_score
    return round(clamp(score, 0.0, 1.0), 4)


def _extract_lane_candidates(
    contours: Sequence[np.ndarray],
    lane_filter: LaneContourFilter,
    frame_id: int,
    timestamp_ms: int,
    roi_shape: tuple[int, int],
    gray: np.ndarray,
    reject_counts: dict | None = None,
    trace: list | None = None,
) -> list[LaneCandidate]:
    """
    Accepted candidates, not yet merged. Every contour lands in exactly one
    reject_counts bucket; trace, if given, gets each contour with the gate that decided it.
    """
    candidates = []
    roi_h, roi_w = roi_shape

    rc = reject_counts if reject_counts is not None else {}
    for _k in ("seen", "area", "degenerate", "too_few_pts",
                "aspect", "w_span", "h_span", "intensity", "accepted"):
        rc.setdefault(_k, 0)

    def reject(gate, contour, value=None):
        rc[gate] += 1
        if trace is not None:
            trace.append({"contour": contour, "bbox": cv2.boundingRect(contour),
                          "gate": gate, "value": value})

    for contour in contours:
        rc["seen"] += 1

        area = cv2.contourArea(contour)
        if area < lane_filter.min_area or area > lane_filter.max_area:
            reject("area", contour, round(area, 1))
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            reject("degenerate", contour)
            continue
        if len(contour) < 5:
            reject("too_few_pts", contour, len(contour))
            continue

        _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        long_side = max(rect_w, rect_h)
        raw_short = min(rect_w, rect_h)
        short_side = max(raw_short, 1.0)
        elongation = long_side / short_side
        if elongation < lane_filter.min_aspect or elongation > lane_filter.max_aspect:
            reject("aspect", contour, round(elongation, 1))
            continue

        horizontal = w >= h
        if horizontal and (w / roi_w) > lane_filter.max_roi_span:
            reject("w_span", contour, round(w / roi_w, 2))
            continue
        if not horizontal and (h / roi_h) > lane_filter.max_roi_span:
            reject("h_span", contour, round(h / roi_h, 2))
            continue

        mean_intensity = _mean_contour_intensity(gray, contour)
        if mean_intensity < lane_filter.min_intensity:
            reject("intensity", contour, round(mean_intensity, 1))
            continue

        confidence = _lane_confidence(
            long = long_side,
            short = raw_short,
            horizontal = horizontal,
            mean_intensity = mean_intensity,
            roi_h = roi_h,
            roi_w = roi_w,
            f = lane_filter
        )

        proximity = clamp((y + h) / max(roi_h, 1), 0.0, 1.0)

        rc["accepted"] += 1
        if trace is not None:
            trace.append({"contour": contour, "bbox": (x, y, w, h), "gate": None, "value": None})
        candidates.append(LaneCandidate(
            label = LANE_BOUNDARY,
            bbox = (x, y, w, h),
            contour = contour,
            confidence = round(confidence, 4),
            length_px = round(long_side, 2),
            width_px = round(raw_short, 2),
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
            proximity = round(proximity, 4),
            mean_intensity = mean_intensity,
            foot_x = contour_foot_x(contour),
        ))

    return candidates


def _filter_lane_contours(
    contours: Sequence[np.ndarray],
    lane_filter: LaneContourFilter,
    frame_id: int,
    timestamp_ms: int,
    roi_shape: tuple[int, int],
    gray: np.ndarray,
) -> list[LaneCandidate]:
    """Production twin of _extract_lane_candidates(), without the counts."""
    candidates = []
    roi_h, roi_w = roi_shape

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < lane_filter.min_area or area > lane_filter.max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            continue
        if len(contour) < 5:
            continue

        _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        long_side = max(rect_w, rect_h)
        raw_short = min(rect_w, rect_h)
        short_side = max(raw_short, 1.0)
        elongation = long_side / short_side
        if elongation < lane_filter.min_aspect or elongation > lane_filter.max_aspect:
            continue

        horizontal = w >= h
        if horizontal and (w / roi_w) > lane_filter.max_roi_span:
            continue
        if not horizontal and (h / roi_h) > lane_filter.max_roi_span:
            continue

        mean_intensity = _mean_contour_intensity(gray, contour)
        if mean_intensity < lane_filter.min_intensity:
            continue

        confidence = _lane_confidence(
            long = long_side,
            short = raw_short,
            horizontal = horizontal,
            mean_intensity = mean_intensity,
            roi_h = roi_h,
            roi_w = roi_w,
            f = lane_filter
        )

        proximity = clamp((y + h) / max(roi_h, 1), 0.0, 1.0)

        candidates.append(LaneCandidate(
            label = LANE_BOUNDARY,
            bbox = (x, y, w, h),
            contour = contour,
            confidence = round(confidence, 4),
            length_px = round(long_side, 2),
            width_px = round(raw_short, 2),
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
            proximity = round(proximity, 4),
            mean_intensity = mean_intensity,
            foot_x = contour_foot_x(contour),
        ))

    return candidates


def extract_lane_candidates(
    lane_roi: np.ndarray,
    canny_params: CannyParams,
    lane_filter: LaneContourFilter,
    frame_id: int,
    timestamp_ms: int,
    draw_overlays: bool = True,
    edges_raw: np.ndarray | None = None,
    horizontal_split: tuple[np.ndarray, np.ndarray] | None = None,
    trace: bool = False,
) -> tuple[list[LaneCandidate], dict]:
    """
    (merged candidates, debug). edges_raw and horizontal_split, when the caller has
    them, save recomputing; the stop-line detector shares them.
    """
    if edges_raw is None:
        edges_raw = _canny(lane_roi, canny_params)
    edges_lane = _strip_horizontal_lines(lane_roi, edges_raw, lane_filter, horizontal_split)
    edges = _close_edges(edges_lane, canny_params.close_kernel)
    contours = _contours(edges)
    reject_counts = {}
    lane_trace = [] if trace else None

    candidates = _extract_lane_candidates(
        contours,
        lane_filter,
        frame_id,
        timestamp_ms,
        lane_roi.shape[:2],
        lane_roi,
        reject_counts,
        lane_trace,
    )

    candidates = _merge_collinear(candidates, lane_filter, lane_roi.shape[:2])
    reject_counts["merged_into"] = len(candidates)

    debug_images = {
        "lane_roi": lane_roi,
        "edges": edges,
        "edges_raw": edges_raw,
        "edges_lane": edges_lane,
        "reject_counts": reject_counts,
    }
    if trace:
        debug_images["trace"] = lane_trace
    if draw_overlays:
        contour_overlay = cv2.cvtColor(lane_roi, cv2.COLOR_GRAY2BGR)
        accepted_overlay = cv2.cvtColor(lane_roi, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(
            contour_overlay,
            contours,
            -1,
            (200, 200, 200),
            1
        )

        for c in candidates:
            cv2.drawContours(
                accepted_overlay,
                [c.contour],
                -1,
                (0, 255, 0),
                2
            )
            x, y, w, h = c.bbox
            cv2.rectangle(
                accepted_overlay,
                (x, y),
                (x + w - 1, y + h - 1),
                (0, 200, 0),
                2
            )
            cv2.putText(
                accepted_overlay,
                f"{c.confidence:.2f}/{c.proximity:.2f}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                cv2.LINE_AA
            )

        debug_images["contour_overlay"] = contour_overlay
        debug_images["accepted_overlay"] = accepted_overlay

    return candidates, debug_images


def find_lane_candidates(
    lane_roi: np.ndarray,
    canny_params: CannyParams,
    lane_filter: LaneContourFilter,
    frame_id: int,
    timestamp_ms: int,
    edges_raw: np.ndarray | None = None,
    horizontal_split: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[LaneCandidate]:
    """Production twin of extract_lane_candidates(), without the debug dict."""
    if edges_raw is None:
        edges_raw = _canny(lane_roi, canny_params)
    edges_lane = _strip_horizontal_lines(lane_roi, edges_raw, lane_filter, horizontal_split)
    edges = _close_edges(edges_lane, canny_params.close_kernel)

    candidates = _filter_lane_contours(
        _contours(edges),
        lane_filter,
        frame_id,
        timestamp_ms,
        lane_roi.shape[:2],
        lane_roi,
    )

    return _merge_collinear(candidates, lane_filter, lane_roi.shape[:2])


def _strip_horizontal_lines(
        lane_roi: np.ndarray,
        edges_raw: np.ndarray,
        lane_filter: LaneContourFilter,
        split: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
    """
    The lane detector's edges without the lines across the lane, which would otherwise
    close into one H-shaped contour with the lane lines and lose the lane. Long
    horizontal runs go, with their stubs within horizontal_band_px; a tape's own
    short ends stay. edges_raw itself when nothing is removed.
    """
    if lane_filter.horizontal_edge_deg is None:
        return edges_raw
    if split is None:
        split = _horizontal_edges(lane_roi, edges_raw, lane_filter.horizontal_edge_deg)
    horizontal = cv2.bitwise_or(split[0], split[1])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    long_runs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= lane_filter.horizontal_min_run_px]
    if not long_runs:
        return edges_raw
    ys, xs = np.nonzero(horizontal)
    drop = np.zeros(len(xs), bool)
    for i in long_runs:
        x, y, w, h = stats[i, :4]
        py, px = np.nonzero(labels[y:y + h, x:x + w] == i)
        vx, vy, x0, y0 = cv2.fitLine(np.stack([px + x, py + y], 1).astype(np.float32),
                                     cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        drop |= np.abs((xs - x0) * vy - (ys - y0) * vx) <= lane_filter.horizontal_band_px
    out = edges_raw.copy()
    out[ys[drop], xs[drop]] = 0
    return out
