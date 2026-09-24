"""Geometry branch: lane boundaries and stop-sign shapes from gray ROIs.

Purpose:
    Answers two structural questions per frame. Lane boundaries: white tape
    on a dark mat gives intensity edges that Canny finds, and contours are
    kept when their area, elongation, span and brightness match tape. Stop
    sign: polygon approximation reduces a contour to its dominant vertices,
    an octagon gives about 8, and area and solidity reject noise. Both share
    the Canny-contour structure but run on separate ROIs.

Main package:
    GeometryBranchResult: one frame's accepted LaneCandidates and
    SignCandidates, all coordinates ROI-relative, with the frame identity
    carried from ROICropResult. Consumed by feature fusion.

Flow:
    1. Validate that both ROIs are single-channel uint8.
    2. Lane: Canny, close along-line gaps, filter contours, merge fragments.
    3. Sign: Canny, filter contours by area, vertex count and solidity.
    4. Package both candidate lists with the frame identity.
"""
import cv2
import numpy as np
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.params import FOOT_BAND_PX, LANE_BOUNDARY, STOP_SIGN
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

@dataclass
class SignContourFilter:
    """Acceptance gates and confidence references for sign-shape contours."""
    min_area: float = 200.0         # px^2
    max_area: float = 30000.0       # px^2
    min_vertices: int = 8           # approxPolyDP vertex count
    max_vertices: int = 10
    min_solidity: float = 0.80      # contour area / convex hull area
    # approxPolyDP epsilon as a fraction of arc length. Smaller keeps more
    # vertices; larger collapses the outline toward fewer.
    epsilon_factor: float = 0.03
    ref_area: float = 5000.0        # px^2 that scores full area confidence

@dataclass(frozen=True)
class GeometryConfig:
    """
    Geometry tuning as one unit, so the stage takes a single config like every
    other stage. The parts can still be passed to run_geometry_branch() directly.
    """
    canny: CannyParams = field(default_factory=CannyParams)     # shared by both branches
    lane: LaneContourFilter = field(default_factory=LaneContourFilter)
    sign: SignContourFilter = field(default_factory=SignContourFilter)

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
class GeometryBranchResult:
    """One frame's geometry detections, ROI-relative. Identity copied from ROICropResult."""
    lane_candidates: list[LaneCandidate]
    sign_candidates: list[SignCandidate]
    frame_id: int
    timestamp_ms: int


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
    reject_counts: dict | None = None
) -> list[LaneCandidate]:
    """
    Run contours through the lane gates and build a candidate for each survivor.

    Inputs:
        roi_shape: (h, w) of the lane ROI; normalizes span and proximity.
        gray: The lane ROI, sampled by the intensity gate.
        reject_counts: Filled with one count per gate. Every contour lands in
            exactly one bucket, so the buckets sum to "seen".

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

    for contour in contours:
        rc["seen"] += 1

        area = cv2.contourArea(contour)
        if area < lane_filter.min_area or area > lane_filter.max_area:
            rc["area"] += 1
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            rc["degenerate"] += 1
            continue
        if len(contour) < 5:
            rc["too_few_pts"] += 1
            continue

        _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        long_side = max(rect_w, rect_h)
        raw_short = min(rect_w, rect_h)
        short_side = max(raw_short, 1.0)
        elongation = long_side / short_side
        if elongation < lane_filter.min_aspect or elongation > lane_filter.max_aspect:
            rc["aspect"] += 1
            continue

        horizontal = w >= h
        if horizontal and (w / roi_w) > lane_filter.max_roi_span:
            rc["w_span"] += 1
            continue
        if not horizontal and (h / roi_h) > lane_filter.max_roi_span:
            rc["h_span"] += 1
            continue

        mean_intensity = _mean_contour_intensity(gray, contour)
        if mean_intensity < lane_filter.min_intensity:
            rc["intensity"] += 1
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
    draw_overlays: bool = True
) -> tuple[list[LaneCandidate], dict]:
    """
    Find lane-boundary candidates in the lane ROI.

    Inputs:
        lane_roi: (h, w) uint8 gray.
        draw_overlays: Also build contour_overlay and accepted_overlay. False
            skips two full-ROI allocations and a contour rasterization per frame.

    Outputs:
        (candidates, debug). Candidates are merged and ROI-relative. debug
        always holds lane_roi, edges, edges_raw and reject_counts (including
        merged_into, the post-merge count).
    """
    edges_raw = _canny(lane_roi, canny_params)
    edges = _close_edges(edges_raw, canny_params.close_kernel)
    contours = _contours(edges)
    reject_counts = {}

    candidates = _extract_lane_candidates(
        contours,
        lane_filter,
        frame_id,
        timestamp_ms,
        lane_roi.shape[:2],
        lane_roi,
        reject_counts
    )

    candidates = _merge_collinear(candidates, lane_filter, lane_roi.shape[:2])
    reject_counts["merged_into"] = len(candidates)

    # @TODO move debug output to a dedicated debug operation
    debug_images = {
        "lane_roi": lane_roi,
        "edges": edges,
        "edges_raw": edges_raw,
        "reject_counts": reject_counts,
    }
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
) -> tuple[GeometryBranchResult, dict, dict]:
    """
    Run lane and sign detection on loose ROI arrays.

    Purpose:
        The entry point for tuning work. The pipeline calls run_geometry_stage().

    Inputs:
        lane_roi, sign_roi: (h, w) uint8 gray, e.g. from ROICropResult.
        draw_overlays: Build the debug overlays in both branches. False skips
            four full-ROI allocations and two contour rasterizations per frame.
        trace: Record the per-contour sign trace (see extract_sign_candidates).

    Outputs:
        (result, lane_debug, sign_debug).

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

    lane_candidates, lane_debug = extract_lane_candidates(
        lane_roi,
        canny_params,
        lane_filter,
        frame_id,
        timestamp_ms,
        draw_overlays
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
        timestamp_ms
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
    )