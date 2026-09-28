"""Geometry branch: lane boundaries, stop lines and stop-sign shapes from gray ROIs.

Purpose:
    Answers three structural questions per frame. Lane boundaries: white tape
    on a dark mat gives intensity edges that Canny finds, and contours are
    kept when their area, elongation, span and brightness match tape. Stop
    lines: the same lane-ROI Canny edges, split by gradient direction so only
    edges of near-horizontal lines remain, then the tape's top and bottom
    edges paired into one line with its own gates. Stop sign: polygon
    approximation reduces a contour to its dominant vertices, an octagon
    gives about 8, and area and solidity reject noise.

    Stop lines only share the Canny result. They have their own config
    (StopLineFilter) and never change what the lane and sign detectors see,
    so lane and sign results are identical with or without them. Geometry
    only detects; the distance to a stop line is stop_line_distance's job.

Main package:
    GeometryBranchResult: one frame's accepted LaneCandidates,
    SignCandidates and StopLineCandidates, all coordinates ROI-relative, with
    the frame identity carried from ROICropResult. Consumed by feature
    fusion, lane_offset and stop_line_distance.

Flow:
    1. Validate that both ROIs are single-channel uint8.
    2. Canny on the lane ROI, once; lanes and stop lines both read it.
    3. Lane: take lines lying across the lane out of the edges (the stop
       line, so it can't close into one contour with the lane lines), close
       along-line gaps, filter contours, merge fragments.
    4. Stop line: keep near-horizontal edges, split them into top (dark to
       bright going down) and bottom edges, fit each, pair top with bottom,
       gate by length, tilt, thickness and brightness.
    5. Sign: Canny on the sign ROI, filter contours by area, vertex count and solidity.
    6. Package the three candidate lists with the frame identity.
"""
import math

import cv2
import numpy as np
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.params import FOOT_BAND_PX, LANE_BOUNDARY, STOP_LINE, STOP_SIGN
from src.utils import clamp
from src.perception.roi_crop import ROICropResult


@dataclass
class CannyParams:
    """Edge detection settings. These set how many contours reach the filters."""
    threshold1: float = 80.0        # hysteresis low; raising it fractures lines into more pieces
    threshold2: float = 200.0       # hysteresis high; raising it seeds fewer contours
    # Sobel aperture: 3, 5 or 7. Larger smooths more but scales gradient
    # magnitudes up, so the thresholds need retuning with it.
    aperture_size: int = 3
    # (width, height) px of the closing rectangle, lane branch only. Wider than
    # tall bridges gaps along horizontal lines. Width < 2 disables closing.
    close_kernel: tuple[int, int] = (9, 3)

@dataclass
class LaneContourFilter:
    """Acceptance gates and confidence references for lane-boundary contours."""
    min_area: float = 1.0           # px^2
    max_area: float = 1000.0        # px^2
    min_aspect: float = 0.0         # elongation: minAreaRect long / short side
    max_aspect: float = 60.0
    max_roi_span: float = 1.0       # max fraction of the ROI a contour may span along its own axis
    min_intensity: float = 120.0    # 0-255 mean inside the contour; rejects dark blobs such as seams
    ref_length: float = 0.25        # long side, as a fraction of ROI extent, that scores full length confidence
    ref_width: float = 30.0         # short side, px, that scores full width confidence
    # Lines across the lane (stop lines) are taken out of the lane detector's
    # edges before contours are traced, so one touching the lane lines can't
    # close into a single contour with them. Edges within this angle of
    # horizontal are the ones considered; None turns the filter off. The
    # course has no curves the camera steers through, so no lane line lies
    # this flat
    horizontal_edge_deg: float | None = 20.0
    # Only horizontal edge runs at least this long are removed, and with them
    # every horizontal edge within horizontal_band_px of their line. Longer
    # than any lane tape is wide (the lane offset gate allows 45 px), so the
    # ends of a piece of tape stay and it still traces as one contour
    horizontal_min_run_px: float = 46.0
    horizontal_band_px: float = 3.0

@dataclass
class SignContourFilter:
    """Acceptance gates and confidence references for sign-shape contours."""
    min_area: float = 100.0         # px^2
    max_area: float = 30000.0       # px^2
    min_vertices: int = 8           # approxPolyDP vertex count
    max_vertices: int = 9
    min_solidity: float = 0.80      # contour area / convex hull area
    # approxPolyDP epsilon as a fraction of arc length. Smaller keeps more
    # vertices; larger collapses the outline toward fewer.
    epsilon_factor: float = 0.03
    ref_area: float = 5000.0        # px^2 that scores full area confidence

@dataclass(frozen=True)
class StopLineFilter:
    """
    Stop-line detection tuning, separate from the lane gates.

    Lengths and thicknesses are lane-ROI px, measured along the fitted line
    and perpendicular to it.
    """
    # Largest angle from horizontal. Also the gradient split: edges of steeper
    # lines never reach the stop-line detector. Covers the robot approaching
    # the line off-square; the course has no curves the camera steers through
    max_tilt_deg: float = 20.0
    # Longer than any lane-tape width (sweep max ~41 px), so the top or bottom
    # edge of a lane-line dash can't pass for a stop line
    min_length_px: float = 60.0
    min_thickness_px: float = 3.0
    max_thickness_px: float = 40.0
    min_intensity: float = 130.0        # 0-255 mean between the paired edges; tape, not a shadow edge
    # Paired edges must overlap by this fraction of the shorter one
    min_edge_overlap: float = 0.5
    # (width, height) px closing rectangle on each edge map; bridges gaps
    # along the line, e.g. where a lane line crosses it
    close_kernel: tuple[int, int] = (15, 1)
    ref_length_px: float = 200.0        # length that scores full length confidence, about one lane

@dataclass(frozen=True)
class GeometryConfig:
    """
    Geometry tuning as one unit, so the stage takes a single config like every
    other stage. The parts can still be passed to run_geometry_branch() directly.
    """
    canny: CannyParams = field(default_factory=CannyParams)     # shared by both branches
    lane: LaneContourFilter = field(default_factory=LaneContourFilter)
    sign: SignContourFilter = field(default_factory=SignContourFilter)
    stop_line: StopLineFilter = field(default_factory=StopLineFilter)   # reads the lane ROI's Canny edges

def contour_foot_x(
        contour: np.ndarray,
        band_px: int = FOOT_BAND_PX,
    ) -> float:
    """
    ROI-local x of a contour at its nearest approach to the robot.

    Purpose:
        The anchor lane offset should steer by. For a marking running
        diagonally across the ROI, the bbox center is displaced from where
        the marking crosses the near edge by up to half the bbox width.

    Inputs:
        contour: OpenCV (N, 1, 2) or flat (N, 2) points.
        band_px: Rows above the lowest point that count as the base. Wider
            averages out a ragged edge; narrower tracks a steep diagonal
            more tightly. 0 or less uses only the lowest row.

    Outputs:
        Mean x of the base points in px, rounded to 0.01. -1.0 for an empty contour.
    """
    if contour is None or len(contour) == 0:
        return -1.0
    pts = np.asarray(contour).reshape(-1, 2)
    y_max = pts[:, 1].max()
    band = pts[pts[:, 1] >= y_max - max(band_px, 0)]
    return round(float(band[:, 0].mean()), 2)


@dataclass
class LaneCandidate:
    """Lane-boundary candidate, ROI-relative. frame_id and timestamp_ms are carried from capture."""
    label: str                          # always "lane_boundary"
    bbox: tuple[int, int, int, int]     # (x, y, w, h)
    contour: np.ndarray                 # (N, 1, 2); a merged candidate holds every member's points
    confidence: float                   # [0, 1] measurement quality, used as a weight
    frame_id: int
    timestamp_ms: int
    proximity: float = 0.0              # [0, 1] bottom-edge position; 1.0 = bottom of ROI, nearest the robot
    width_px: float = 0.0               # minAreaRect short side, px; raw, not floored to 1 like the aspect gate
    length_px: float = 0.0              # minAreaRect long side, px
    mean_intensity: float = 0.0         # 0-255 inside the filled contour; length-weighted when merged
    foot_x: float = -1.0                # see contour_foot_x; -1.0 = not computed

@dataclass
class SignCandidate:
    """Stop-sign candidate, ROI-relative. frame_id and timestamp_ms are carried from capture."""
    label: str                          # always "stop_sign"
    bbox: tuple[int, int, int, int]     # (x, y, w, h) of the raw contour
    contour: np.ndarray                 # the approxPolyDP polygon, not the raw contour
    vertex_count: int                   # == len(contour)
    confidence: float                   # [0, 1]
    frame_id: int
    timestamp_ms: int
    area: float = 0.0                   # raw contour area, px^2; 0.0 = not computed
    solidity: float = 0.0               # area / convex hull area; 0.0 = not computed

@dataclass
class StopLineCandidate:
    """
    Stop-line candidate, lane-ROI-relative. frame_id and timestamp_ms are carried from capture.

    Rows grow downward, toward the robot. y_near_px is the lowest point of the
    bottom edge: the part of the line the robot reaches first.
    """
    label: str                          # always "stop_line"
    bbox: tuple[int, int, int, int]     # (x, y, w, h) around the paired edges
    x_left: float                       # ends of the overlap of the paired edges, px
    x_right: float
    y_top_px: float                     # top edge's row at the midpoint of the overlap
    y_bottom_px: float                  # bottom edge's row at the midpoint; the ROI height when clipped
    y_near_px: float                    # lowest bottom-edge row across the overlap
    tilt_deg: float                     # signed angle from horizontal; + = right end nearer the robot
    length_px: float                    # along the line
    thickness_px: float                 # perpendicular to the line
    mean_intensity: float               # 0-255 between the edges
    clipped: bool                       # the bottom edge is below the ROI: the robot is on the line
    confidence: float                   # [0, 1]
    frame_id: int
    timestamp_ms: int
    proximity: float = 0.0              # [0, 1] y_near_px / ROI height, as LaneCandidate's; 1.0 = at the ROI bottom

@dataclass
class GeometryBranchResult:
    """One frame's geometry detections, ROI-relative. Identity copied from ROICropResult."""
    lane_candidates: list[LaneCandidate]
    sign_candidates: list[SignCandidate]
    frame_id: int
    timestamp_ms: int
    stop_line_candidates: list[StopLineCandidate] = field(default_factory=list)   # lane-ROI-relative, nearest first


def _close_edges(
        edges: np.ndarray,
        kernel_size: tuple[int, int]
    ) -> np.ndarray:
    """Bridge along-line gaps in an edge map so a fragmented lane line traces as one contour."""
    if not kernel_size or kernel_size[0] < 2:
        return edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

def _canny(
        gray: np.ndarray,
        params: CannyParams
    ) -> np.ndarray:
    """0/255 Canny edge map of a gray image."""
    return cv2.Canny(gray, params.threshold1, params.threshold2,
                        apertureSize=params.aperture_size)

def _contours(
        edges: np.ndarray
    ) -> Sequence[np.ndarray]:
    """External contours of an edge map."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def _extreme_points(
        contour: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """Leftmost and rightmost [x, y] of a contour; fragment merging measures the gap between them."""
    pts = contour.reshape(-1, 2)
    return pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()]


# @TODO integrate into _extract_lane_candidates. Kept separate so it can be tested in isolation.
def _mean_contour_intensity(
        gray: np.ndarray,
        contour: np.ndarray
    ) -> float:
    """Mean gray level inside the filled contour, or 0.0 if it lies entirely off the image."""
    x, y, w, h = cv2.boundingRect(contour)

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, gray.shape[1]), min(y + h, gray.shape[0])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi_patch = gray[y1:y2, x1:x2]
    mask = np.zeros(roi_patch.shape, dtype=np.uint8)
    # The mask is patch-sized, so move the contour into patch coordinates
    shifted = contour - np.array([[[x1, y1]]])
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    pixels = roi_patch[mask == 255]
    return float(np.mean(pixels)) if len(pixels) > 0 else 0.0

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

def _sign_confidence(
        area: float,
        vertex_count: int,
        f: SignContourFilter
    ) -> float:
    """Equal blend of closeness to 8 vertices (an octagon) and area up to ref_area, in [0, 1]."""
    vertex_score = clamp(1.0 - abs(vertex_count - 8) / 8.0, 0.0, 1.0)
    denom = max(f.ref_area - f.min_area, 1.0)
    area_score = clamp((area - f.min_area) / denom, 0.0, 1.0)
    return round(0.5 * vertex_score + 0.5 * area_score, 4)


# An area-rejected contour is traced only if its bounding box covers at least
# this fraction of min_area; smaller ones are edge noise and are only counted.
# Judged by bbox, not contour area: a Canny outline with a gap traces as a thin
# sliver with almost no enclosed area but a sign-sized box, and that is exactly
# the failure the trace is there to show
TRACE_MIN_BBOX_FRAC = 1.0

def _trace_entry(
        contour: np.ndarray,
        gate: str | None,
        area: float,
        vertices: int | None = None,
        solidity: float | None = None,
        confidence: float | None = None,
        poly: np.ndarray | None = None,
    ) -> dict:
    """One sign-trace row: bbox plus the gate that decided (None = accepted); unmeasured fields stay None."""
    return {
        "bbox": cv2.boundingRect(contour),
        "gate": gate,
        "area": area,
        "vertices": vertices,
        "solidity": solidity,
        "confidence": confidence,
        "poly": poly,
    }

def _extract_sign_candidates(
    contours: Sequence[np.ndarray],
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
    reject_counts: dict | None = None,
    trace: list[dict] | None = None,
) -> list[SignCandidate]:
    """
    Run contours through the sign gates: area, vertex count, convex hull, solidity.

    Inputs:
        reject_counts: Filled with one count per gate. Every contour lands in
            exactly one bucket, so the buckets sum to "seen".
        trace: A list to receive one _trace_entry() per contour that reached a
            gate, accepted or not, or None to skip tracing. Area rejects are
            traced only when their bbox clears TRACE_MIN_BBOX_FRAC.

    Outputs:
        Accepted candidates, each holding its approxPolyDP polygon.

    Side effects:
        Mutates reject_counts and trace.
    """
    candidates = []

    rc = reject_counts if reject_counts is not None else {}
    for _k in ("seen", "area", "vertices", "hull", "solidity", "accepted"):
        rc.setdefault(_k, 0)

    for contour in contours:
        rc["seen"] += 1
        area = cv2.contourArea(contour)
        if area < sign_filter.min_area or area > sign_filter.max_area:
            rc["area"] += 1
            if trace is not None:
                _, _, bw, bh = cv2.boundingRect(contour)
                if bw * bh >= sign_filter.min_area * TRACE_MIN_BBOX_FRAC:
                    trace.append(_trace_entry(contour, "area", area))
            continue

        arc_len = cv2.arcLength(contour, closed=True)
        epsilon = sign_filter.epsilon_factor * arc_len
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        n_verts = len(approx)
        confidence = _sign_confidence(area, n_verts, sign_filter)

        if n_verts < sign_filter.min_vertices or n_verts > sign_filter.max_vertices:
            rc["vertices"] += 1
            if trace is not None:
                trace.append(_trace_entry(
                    contour, "vertices", area, n_verts,
                    confidence=confidence, poly=approx))
            continue

        # Reject non-convex / fragmented shapes
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            rc["hull"] += 1
            if trace is not None:
                trace.append(_trace_entry(
                    contour, "hull", area, n_verts,
                    confidence=confidence, poly=approx))
            continue
        solidity = area / hull_area
        if solidity < sign_filter.min_solidity:
            rc["solidity"] += 1
            if trace is not None:
                trace.append(_trace_entry(
                    contour, "solidity", area, n_verts, round(solidity, 4),
                    confidence, approx))
            continue

        x, y, w, h = cv2.boundingRect(contour)
        rc["accepted"] += 1

        candidates.append(SignCandidate(
            label = STOP_SIGN,
            bbox = (x, y, w, h),
            contour = approx,
            vertex_count = n_verts,
            confidence = confidence,
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
            area = area,
            solidity = round(solidity, 4),
        ))
        if trace is not None:
            trace.append(_trace_entry(
                contour, None, area, n_verts, round(solidity, 4),
                confidence, approx))

    return candidates

def _filter_sign_contours(
    contours: Sequence[np.ndarray],
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
) -> list[SignCandidate]:
    """
    Production twin of _extract_sign_candidates(): same gates in the same
    order, without reject counting or tracing.

    Outputs:
        Accepted candidates, each holding its approxPolyDP polygon.
    """
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < sign_filter.min_area or area > sign_filter.max_area:
            continue

        arc_len = cv2.arcLength(contour, closed=True)
        epsilon = sign_filter.epsilon_factor * arc_len
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        n_verts = len(approx)
        if n_verts < sign_filter.min_vertices or n_verts > sign_filter.max_vertices:
            continue

        # Reject non-convex / fragmented shapes
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < sign_filter.min_solidity:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        candidates.append(SignCandidate(
            label = STOP_SIGN,
            bbox = (x, y, w, h),
            contour = approx,
            vertex_count = n_verts,
            confidence = _sign_confidence(area, n_verts, sign_filter),
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
            area = area,
            solidity = round(solidity, 4),
        ))

    return candidates

def extract_sign_candidates(
    sign_roi: np.ndarray,
    canny_params: CannyParams,
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
    draw_overlays: bool = True,
    trace: bool = False,
) -> tuple[list[SignCandidate], dict]:
    """
    Find stop-sign-shaped candidates in the sign ROI.

    Inputs:
        sign_roi: (h, w) uint8 gray.
        draw_overlays: Also build contour_overlay and accepted_overlay. False
            skips two full-ROI allocations and a contour rasterization per frame.
        trace: Record every contour that reached a gate, and the gate that
            decided it, under debug["trace"]. Off by default; the live loop
            doesn't read it.

    Outputs:
        (candidates, debug). debug always holds sign_roi, edges and
        reject_counts. With trace, debug["trace"] is a list of dicts (bbox,
        gate, area, vertices, solidity, confidence, poly), ROI-relative; gate
        is None for accepted candidates.
    """
    edges = _canny(sign_roi, canny_params)
    contours = _contours(edges)

    reject_counts = {}
    trace_log = [] if trace else None

    candidates = _extract_sign_candidates(
        contours,
        sign_filter,
        frame_id,
        timestamp_ms,
        reject_counts,
        trace_log,
    )

    debug_images = {
        "sign_roi": sign_roi,
        "edges": edges,
        "reject_counts": reject_counts,
    }
    if trace:
        debug_images["trace"] = trace_log

    if draw_overlays:
        # Both overlays must be 3-channel: a single-channel destination keeps
        # only the first BGR component, so every annotation would render black
        contour_overlay = cv2.cvtColor(sign_roi, cv2.COLOR_GRAY2BGR)
        accepted_overlay = cv2.cvtColor(sign_roi, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(contour_overlay, contours, -1, (200, 200, 200), 1)
        for c in candidates:
            cv2.drawContours(accepted_overlay, [c.contour], -1, (0, 0, 255), 2)
            x, y, w, h = c.bbox
            cv2.rectangle(accepted_overlay, (x, y), (x + w - 1, y + h - 1), (0, 0, 200), 1)
            cv2.putText(accepted_overlay, f"v={c.vertex_count} {c.confidence:.2f}",
                        (x, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

        debug_images["contour_overlay"] = contour_overlay
        debug_images["accepted_overlay"] = accepted_overlay

    return candidates, debug_images


def find_sign_candidates(
    sign_roi: np.ndarray,
    canny_params: CannyParams,
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
) -> list[SignCandidate]:
    """
    Production twin of extract_sign_candidates(): no debug dict, overlays or trace.

    Outputs:
        ROI-relative candidates.
    """
    edges = _canny(sign_roi, canny_params)

    return _filter_sign_contours(
        _contours(edges),
        sign_filter,
        frame_id,
        timestamp_ms,
    )

# =============================================================================
# Stop lines
# =============================================================================

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
        """Signed angle from horizontal; + = the right end is lower (nearer the robot)."""
        return math.degrees(math.atan(self.slope))

    @property
    def length_px(self) -> float:
        return (self.x1 - self.x0) / math.cos(math.atan(self.slope))

def _fit_segment(
        points: np.ndarray,
    ) -> _EdgeSegment | None:
    """Fit a line through edge points; None if they span under 2 px in x (not a horizontal edge)."""
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    if x1 - x0 < 2.0:
        return None
    vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    if abs(vx) < 1e-6:
        return None
    return _EdgeSegment(points, x0, x1, float(cx), float(cy), float(vy / vx))

def _horizontal_edges(
        lane_roi: np.ndarray,
        edges_raw: np.ndarray,
        max_tilt_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the lane ROI's Canny edges by gradient direction.

    Purpose:
        A stop line's top and bottom edges have a gradient pointing up or
        down; a lane line's sides have one pointing left or right. Keeping
        only edge pixels whose line runs within max_tilt_deg of horizontal
        separates a stop line from lane lines it touches, before any contour
        joins them.

    Outputs:
        (top, bottom): 0/255 uint8 maps. top holds edges where brightness
        rises going down (the upper edge of bright tape), bottom where it falls.
    """
    gx = cv2.Sobel(lane_roi, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(lane_roi, cv2.CV_16S, 0, 1, ksize=3)
    # Only edge pixels are classified: a few hundred of the ROI's ~35k, so
    # whole-array arithmetic would cost ten times the work it needs
    ys, xs = np.nonzero(edges_raw)
    ex = np.abs(gx[ys, xs].astype(np.int32))
    ey = gy[ys, xs].astype(np.int32)
    # |gx| <= |gy| tan(tilt), in integers: the edge's line is within tilt of horizontal
    scale = 1024
    horizontal = ex * scale <= np.abs(ey) * int(round(math.tan(math.radians(max_tilt_deg)) * scale))
    top = np.zeros_like(edges_raw)
    bottom = np.zeros_like(edges_raw)
    rising = horizontal & (ey > 0)
    falling = horizontal & (ey < 0)
    top[ys[rising], xs[rising]] = 255
    bottom[ys[falling], xs[falling]] = 255
    return top, bottom

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

def _edge_segments(
        mask: np.ndarray,
        f: StopLineFilter,
    ) -> list[_EdgeSegment]:
    """
    Near-horizontal edges of one polarity as fitted segments, with pieces of one edge joined.

    Purpose:
        A lane line crossing a stop line breaks the stop line's edge for the
        width of the lane tape. Pieces whose facing ends are within
        max_thickness_px in x and whose fitted lines agree within 3 px there
        are refit as one edge.

    Outputs:
        Segments sorted by x0.
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
    """[0, 1], rounded to 4 places: 50% length, 30% intensity margin above min_intensity, 20% squareness."""
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
    Gate one top edge and pair it with the bottom edge of the same tape.

    Purpose:
        Shared by the debug and production detectors, so their gates can't
        drift; the debug one counts the returned reason, the production one
        ignores it.

    Inputs:
        top: A top edge (brightness rising going down).
        bottoms: Every bottom edge in the ROI.

    Outputs:
        (candidate, "accepted"), or (None, reason) with reason one of
        "short", "tilt", "unpaired", "intensity". Unpaired means no bottom
        edge overlaps it at a plausible thickness, and it isn't close enough
        to the ROI bottom to be a line the robot is already on.
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
        # The robot may already be on the line: its bottom edge is below the ROI
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
    """
    Find stop lines in the lane ROI from its Canny edges.

    Inputs:
        lane_roi: (h, w) uint8 gray.
        edges_raw: Canny of lane_roi, the same map the lane detector reads.
        split: _horizontal_edges at stop_filter.max_tilt_deg, if the caller
            has it; computed here otherwise.

    Outputs:
        (candidates, debug). Candidates are lane-ROI-relative, nearest the
        robot first. debug holds edges_top and edges_bottom (the gradient
        split), top_count, bottom_count, reject_counts per gate, and for the
        stop-line view: trace, one entry per top edge ({"ends": ((x0, y0),
        (x1, y1)) of its fitted line, "gate": None or the rejecting gate,
        "length", "tilt", "candidate": the StopLineCandidate or None}), and
        bottoms, the fitted ends of every bottom edge.
    """
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
    """
    Production twin of extract_stop_line_candidates(): no debug dict or reject counts.

    Outputs:
        Lane-ROI-relative candidates, nearest the robot first.
    """
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


def _splits(lane_roi, lane_edges, lane_filter: LaneContourFilter, stop_filter: StopLineFilter):
    """
    The gradient split each lane-ROI detector needs, computed once when their angles agree.

    Outputs:
        (stop-line split, lane split); the lane split is None when the lane
        filter is off or uses another angle (it then computes its own).
    """
    stop_split = _horizontal_edges(lane_roi, lane_edges, stop_filter.max_tilt_deg)
    lane_split = stop_split if lane_filter.horizontal_edge_deg == stop_filter.max_tilt_deg else None
    return stop_split, lane_split


def run_geometry_branch(
    lane_roi: np.ndarray,
    sign_roi: np.ndarray,
    canny_params: CannyParams,
    lane_filter: LaneContourFilter,
    sign_filter: SignContourFilter,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    draw_overlays: bool = True,
    trace: bool = False,
    stop_filter: StopLineFilter = StopLineFilter(),
) -> tuple[GeometryBranchResult, dict, dict]:
    """
    Run lane, stop-line and sign detection on loose ROI arrays.

    Purpose:
        The entry point for tuning work. The pipeline calls run_geometry_stage().

    Inputs:
        lane_roi, sign_roi: (h, w) uint8 gray, e.g. from ROICropResult.
        draw_overlays: Build the debug overlays in both branches. False skips
            four full-ROI allocations and two contour rasterizations per frame.
        trace: Record the per-contour sign trace (see extract_sign_candidates)
            and the lane contour trace (see extract_lane_candidates).
        stop_filter: Stop-line gates. Stop lines read the lane ROI's Canny
            edges and never change the lane or sign results.

    Outputs:
        (result, lane_debug, sign_debug). lane_debug["stop_line"] holds the
        stop-line detector's debug (see extract_stop_line_candidates).

    Raises:
        ValueError / TypeError: If either ROI is None, not uint8, or not 2-D.
            The message names the offending ROI.
    """
    for name, roi in [("lane_roi", lane_roi), ("sign_roi", sign_roi)]:
        if roi is None:
            raise ValueError(
                f"run_geometry_branch: {name} is None"
            )
        if roi.dtype != np.uint8:
            raise TypeError(
                f"run_geometry_branch: {name} expected uint8, got {roi.dtype}"
            )
        if roi.ndim != 2:
            raise ValueError(
                f"run_geometry_branch: {name} expected single-channel grayscale "
                f"(H,W), got {roi.shape}. Color conversion belongs to preprocess."
            )

    lane_edges = _canny(lane_roi, canny_params)     # shared by lanes and stop lines
    stop_split, lane_split = _splits(lane_roi, lane_edges, lane_filter, stop_filter)

    lane_candidates, lane_debug = extract_lane_candidates(
        lane_roi,
        canny_params,
        lane_filter,
        frame_id,
        timestamp_ms,
        draw_overlays,
        edges_raw = lane_edges,
        horizontal_split = lane_split,
        trace = trace,
    )

    stop_line_candidates, lane_debug["stop_line"] = extract_stop_line_candidates(
        lane_roi,
        lane_edges,
        stop_filter,
        frame_id,
        timestamp_ms,
        split = stop_split,
    )

    sign_candidates, sign_debug = extract_sign_candidates(
        sign_roi,
        canny_params,
        sign_filter,
        frame_id,
        timestamp_ms,
        draw_overlays,
        trace
    )

    result = GeometryBranchResult(
        lane_candidates,
        sign_candidates,
        frame_id,
        timestamp_ms,
        stop_line_candidates,
    )

    return result, lane_debug, sign_debug


def run_geometry_stage(
        roi: ROICropResult,
        config: GeometryConfig = GeometryConfig(),
        draw_overlays: bool = False,
        trace: bool = False,
    ) -> tuple[GeometryBranchResult, dict, dict]:
    """
    Stage entry point: run the geometry branch on one ROICropResult.

    Purpose:
        Same shape as preprocess_frame() and crop_rois(), so the orchestrator
        chains single-argument calls instead of re-threading the frame
        identity by hand. A thin adapter over run_geometry_branch().

    Inputs:
        draw_overlays: Off by default; the live loop discards the debug dicts,
            so building them would cost allocations nothing reads. The
            dataset harness passes True.
        trace: Off by default. Records the sign trace for the debug views.

    Outputs:
        (result, lane_debug, sign_debug), with frame_id and timestamp_ms
        carried from the ROICropResult.
    """
    return run_geometry_branch(
        lane_roi = roi.lane_roi,
        sign_roi = roi.sign_roi,
        canny_params = config.canny,
        lane_filter = config.lane,
        sign_filter = config.sign,
        frame_id = roi.frame_id,
        timestamp_ms = roi.timestamp_ms,
        draw_overlays = draw_overlays,
        trace = trace,
        stop_filter = config.stop_line,
    )


def detect_geometry(
        roi: ROICropResult,
        config: GeometryConfig = GeometryConfig(),
    ) -> GeometryBranchResult:
    """
    Production twin of run_geometry_stage(): same stage, result only.

    Outputs:
        GeometryBranchResult, with frame_id and timestamp_ms carried from roi.

    Raises:
        ValueError / TypeError: If either ROI is None, not uint8, or not 2-D.
            The message names the offending ROI.
    """
    for name, r in [("lane_roi", roi.lane_roi), ("sign_roi", roi.sign_roi)]:
        if r is None:
            raise ValueError(
                f"detect_geometry: {name} is None"
            )
        if r.dtype != np.uint8:
            raise TypeError(
                f"detect_geometry: {name} expected uint8, got {r.dtype}"
            )
        if r.ndim != 2:
            raise ValueError(
                f"detect_geometry: {name} expected single-channel grayscale "
                f"(H,W), got {r.shape}. Color conversion belongs to preprocess."
            )

    lane_edges = _canny(roi.lane_roi, config.canny)     # shared by lanes and stop lines
    stop_split, lane_split = _splits(roi.lane_roi, lane_edges, config.lane, config.stop_line)

    lane_candidates = find_lane_candidates(
        roi.lane_roi,
        config.canny,
        config.lane,
        roi.frame_id,
        roi.timestamp_ms,
        edges_raw = lane_edges,
        horizontal_split = lane_split,
    )

    stop_line_candidates = find_stop_line_candidates(
        roi.lane_roi,
        lane_edges,
        config.stop_line,
        roi.frame_id,
        roi.timestamp_ms,
        split = stop_split,
    )

    sign_candidates = find_sign_candidates(
        roi.sign_roi,
        config.canny,
        config.sign,
        roi.frame_id,
        roi.timestamp_ms,
    )

    return GeometryBranchResult(
        lane_candidates,
        sign_candidates,
        roi.frame_id,
        roi.timestamp_ms,
        stop_line_candidates,
    )
