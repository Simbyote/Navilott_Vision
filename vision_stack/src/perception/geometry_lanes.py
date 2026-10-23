"""Lane boundaries from the gray lane ROI: Canny edges, gated contours, merged fragments.

Purpose:
    White tape on a dark mat gives intensity edges that Canny finds;
    contours are kept when their area, elongation, span and brightness match
    tape. Lines lying across the lane (a stop line) come out of this
    detector's copy of the edges first, so they can't close into one contour
    with the lane lines.

Main package:
    LaneCandidate: one accepted lane marking, its bbox, foot_x (where it
    meets the bottom of the view), confidence and contour.

Flow:
    1. Take lines lying across the lane out of the edges.
    2. Close along-line gaps; trace contours.
    3. Gate each contour; score the survivors.
    4. Merge fragments of one line.
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
    """Leftmost and rightmost [x, y] of a contour; fragment merging measures the gap between them."""
    pts = contour.reshape(-1, 2)
    return pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()]


def _merge_collinear(
        candidates: list[LaneCandidate],
        lane_filter: LaneContourFilter,
        roi_shape: tuple[int, int],
        max_gap_px: float = 40.0,
    ) -> list[LaneCandidate]:
    """
    Join fragments of one horizontal lane line into a single candidate.

    Purpose:
        Without this, lane_offset could select two pieces of the same line
        as opposite boundaries.

    Inputs:
        roi_shape: (h, w) of the lane ROI.
        max_gap_px: Largest endpoint gap, px, still treated as the same line
            (inclusive). Larger heals wider breaks but risks joining two
            separate lines.

    Outputs:
        Merged horizontal candidates followed by the vertical ones, which
        pass through untouched. A fragment with no neighbor is returned as is.
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

        # Length-weighted mean of member intensities; refilling a concatenated
        # point set would draw a polygon that is not the line
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
    """
    Score how far a lane candidate's measurements can be trusted.

    Inputs:
        long, short: minAreaRect sides, px.
        horizontal: Whether the marking runs across the ROI. Picks which ROI
            extent the length is normalized against.
        mean_intensity: 0-255 mean inside the contour.

    Outputs:
        [0, 1], rounded to 4 places: 50% length, 30% intensity margin above
        min_intensity, 20% thickness.
    """
    # Normalized against the ROI span in the direction of the contour
    extent = roi_w if horizontal else roi_h
    ref_len = max(f.ref_length * max(extent, 1), 1.0)
    length_score = clamp(long / ref_len, 0.0, 1.0)

    # Thickness separates tape from hairline edge traces
    width_score = clamp(short / max(f.ref_width, 1.0), 0.0, 1.0)

    # Ranked by how far above the acceptance threshold it sits
    denom = max(255.0 - f.min_intensity, 1.0)
    intensity_score = clamp((mean_intensity - f.min_intensity) / denom, 0.0, 1.0)

    score = 0.5 * length_score + 0.3 * intensity_score + 0.2 * width_score
    return round(clamp(score, 0.0, 1.0), 4)


# @TODO split each gate into its own function, the way _mean_contour_intensity is
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
    Run contours through the lane gates and build a candidate for each survivor.

    Inputs:
        roi_shape: (h, w) of the lane ROI; normalizes span and proximity.
        gray: The lane ROI, sampled by the intensity gate.
        reject_counts: Filled with one count per gate. Every contour lands in
            exactly one bucket, so the buckets sum to "seen".
        trace: If given, one entry per contour is appended: {"contour",
            "bbox", "gate" (None when accepted), "value" (what the gate
            measured, or None)}, for the lane-geometry view.

    Outputs:
        Accepted candidates, not yet merged.

    Side effects:
        Mutates reject_counts.
    """
    candidates = []
    roi_h, roi_w = roi_shape

    # @TODO move reject counting into a dedicated debug/logging path
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
    """
    Production twin of _extract_lane_candidates(): same gates in the same
    order, without reject counting.

    Outputs:
        Accepted candidates, not yet merged.
    """
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
    Find lane-boundary candidates in the lane ROI.

    Inputs:
        lane_roi: (h, w) uint8 gray.
        draw_overlays: Also build contour_overlay and accepted_overlay. False
            skips two full-ROI allocations and a contour rasterization per frame.
        edges_raw: Canny of lane_roi with canny_params, when the caller has
            already run it (the stop-line detector shares it). None runs it here.
        horizontal_split: _horizontal_edges(lane_roi, edges_raw,
            lane_filter.horizontal_edge_deg), when the caller has it at that
            angle. None computes it here if the filter is on.
        trace: Record every contour with the gate that decided it in
            debug["trace"] (see _extract_lane_candidates), before merging.

    Outputs:
        (candidates, debug). Candidates are merged and ROI-relative. debug
        always holds lane_roi, edges, edges_raw, edges_lane (edges_raw with
        the lines across the lane taken out, what the contours come from)
        and reject_counts (including merged_into, the post-merge count).
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

    # @TODO move debug output to a dedicated debug operation
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
        # 3-channel so the BGR annotations render (see extract_sign_candidates)
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
            cv2.putText(    # confidence/proximity
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
    """
    Production twin of extract_lane_candidates(): no debug dict, no overlays.

    Inputs:
        edges_raw, horizontal_split: As for extract_lane_candidates().

    Outputs:
        Merged, ROI-relative candidates.
    """
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
    The lane detector's edges, with lines lying across the lane taken out.

    Purpose:
        A stop line touching the lane lines otherwise closes into one
        H-shaped contour with them, too wide for the lane gates, and the
        lane is lost for as long as the line is in view. The stop-line
        detector still reads the full edge map; only the lane detector's copy
        loses them.

        Only long horizontal edge runs go (at least horizontal_min_run_px),
        with every horizontal edge within horizontal_band_px of their fitted
        line, so the stubs of a stop line running past the tape go too. The
        short ends of a piece of tape stay, so it still traces as one contour.

    Inputs:
        split: (top, bottom) from _horizontal_edges at
            lane_filter.horizontal_edge_deg, if the caller has it.

    Outputs:
        edges_raw itself when the filter is off or finds nothing; otherwise a copy.
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
