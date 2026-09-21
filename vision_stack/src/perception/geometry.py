"""
geometry_branch.py

Geometry Branch Stage

Purpose:
    The geometry branch answers two structural questions from the frame:

  1. Where are the lane boundaries?
     Lane markings are white tape on a dark mat; there are intensity discontinuities
     that Canny can detect 
     Contours are filtered by aspect ratio and area to accept long, thin, and roughly 
     horizontal or vertical shapes that match lane line geometry, and reject most background 
     clutter

  2. Is there a stop sign shape present?
     The stop sign is an octagon. Contour approximation reduces a contour to its dominant vertices 
     An octagon produces approximately 8 vertices 
     Area and convexity filtering further discriminate against noise contours

These two detections operate on different ROIs and are logically separated
even though they share the grayscale-Canny-contour pipeline structure
The results of both are returned together for feature fusion

All coordinates are ROI-relative.
@NOTE Think about moving each calculation done in _extract_lane_candidates into its own function
        similar to how _mean_contour_intensity is implemented
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List

from src.perception.roi_crop import ROICropResult

# ============================================================================
# Input Dataclasses
# ============================================================================
@dataclass
class CannyParams:  # Edge Detection
    """
    Parameters for cv2.Canny edge detection --- affects how many contours are extracted (edges.png)

    threshold1: lower hysteresis threshold --- raising creates more fracturing
    threshold2: upper hysteresis threshold --- raising reduces contours
    aperture_size: Sobel kernel size (3, 5, or 7) --- higher kernel size smooths more noise
    close_kernel: kernel size for morphological closing
    """
    threshold1: float = 80.0
    threshold2: float = 200.0
    aperture_size: int = 3
    close_kernel: tuple = (9, 3)

@dataclass
class LaneContourFilter:    # Lane Boundary
    """
    Geometric acceptance criteria for lane-boundary contours

    min_area: The minimum area a detected contour must have
    max_area: The maximum area a detected contour can have
    min_aspect: lower bound of a long/short side from minAreaRect
    max_aspect: upper bound of a long/short side from minAreaRect
    min_intensity: Minimum intensity (0-255) inside the contour. Rejects dark blobs
    ref_length: The contour long side, as a fraction of the ROI extent along that axis
    ref_width: The contour short side, as pixels of the ROI extent along that axis
    """
    min_area: float = 1.0
    max_area: float = 1000.0
    min_aspect: float = 0.0
    max_aspect: float = 60.0
    max_roi_span: float = 1.0
    min_intensity: float = 120.0
    ref_length: float = 0.25
    ref_width: float = 30.0

@dataclass
class SignContourFilter:    # Stop Sign
    """
    Geometric acceptance criteria for sign-shape contours

    min_area: minimum contour area
    max_area: maximum contour area
    min_vertices: approxPolyDP vertex count lower bound
    max_vertices: approxPolyDP vertex count upper bound
    min_solidity: contour_area / convex_hull_area
    epsilon_factor: approxPolyDP epsilon = epsilon_factor * arc_length
                     smaller = more vertices retained; larger = fewer
    ref_area: area treated as confidence area_score = 1.0
    """
    min_area: float = 200.0
    max_area: float = 30000.0
    min_vertices: int = 8
    max_vertices: int = 10
    min_solidity: float = 0.80
    epsilon_factor: float = 0.03
    ref_area: float = 5000.0

@dataclass(frozen=True)
class GeometryConfig:
    """
    The geometry branch's tuning as one unit, so the stage takes a single
    config argument like every other stage

    canny: edge detection parameters, shared by both branches
    lane: lane-boundary contour acceptance criteria
    sign: sign-shape contour acceptance criteria

    The individual dataclasses are unchanged and can still be passed
    directly to run_geometry_branch()
    """
    canny: CannyParams = field(default_factory=CannyParams)
    lane: LaneContourFilter = field(default_factory=LaneContourFilter)
    sign: SignContourFilter = field(default_factory=SignContourFilter)

FOOT_BAND_PX = 6      # height of the band at a contour's base used for foot_x

def contour_foot_x(
        contour: np.ndarray,
        band_px: int = FOOT_BAND_PX,
    ) -> float:
    """
    Purpose:
        ROI-local x of a contour at its nearest approach to the robot: the
        mean x of the points within band_px of its lowest row

    Notes:
        This is the anchor lane offset estimation should steer by. A bbox
        centroid answers "where is the middle of this box", which for a marking
        running diagonally across the ROI is displaced from where the marking
        actually crosses the near edge --- by up to half the bbox width
    """
    if contour is None or len(contour) == 0:
        return -1.0
    pts = np.asarray(contour).reshape(-1, 2)
    y_max = pts[:, 1].max()
    band = pts[pts[:, 1] >= y_max - max(band_px, 0)]
    return round(float(band[:, 0].mean()), 2)

# ============================================================================
# Output Dataclasses
# ===========================================================================
@dataclass
class LaneCandidate:
    """
    Lane boundary candidate

    label: lane boundary type
    bbox: (x, y, w, h) of the candidate's bounding box
    contour: lane boundary contour
    confidence: detection confidence
    frame_id: frame identifier
    timestamp_ms: time at which the detection was made
    proximity: vertical position prior [0,1], 1.0 == bottom of ROI
    width_px: minAreaRect short side in px @NOTE is a raw measurement, 
    length_px: minAreaRect long side in px @NOTE is a raw measurement,
    mean_intensity: mean pixel intensity within the candidate's bounding box
    foot_x: ROI-local x where the marking sits at its nearest approach to the
            robot --- the mean x of the contour points in its lowest rows. The
            bbox centroid is the midpoint of a box, which for an angled marking
            is not where the marking crosses the near edge of the ROI. -1.0
            means not computed
    """
    label: str
    bbox: tuple
    contour: np.ndarray
    confidence: float
    frame_id: int
    timestamp_ms: int
    proximity: float = 0.0
    width_px: float = 0.0
    length_px: float = 0.0
    mean_intensity: float = 0.0
    foot_x: float = -1.0

@dataclass
class SignCandidate:
    """
    Stop sign candidate

    label: stop sign type
    bbox: (x, y, w, h) of the candidate's bounding box
    contour: stop sign contour
    vertex_count: number of vertices in the contour
    confidence: detection confidence
    frame_id: frame identifier
    timestamp_ms: time at which the detection was made
    area: contour area in px^2 (ROI-relative). 0.0 means not computed
    solidity: contour_area / convex_hull_area. 0.0 means not computed
    """
    label: str
    bbox: tuple
    contour: np.ndarray
    vertex_count: int
    confidence: float
    frame_id: int
    timestamp_ms: int
    area: float = 0.0
    solidity: float = 0.0

@dataclass
class GeometryBranchResult:
    """
    Output of the geometry branch

    lane_candidates: list of lane boundary candidates
    sign_candidates: list of stop sign candidates
    frame_id: frame identifier
    timestamp_ms: time at which the detection was made
    """
    lane_candidates: List[LaneCandidate]
    sign_candidates: List[SignCandidate]
    frame_id: int
    timestamp_ms: int

# ============================================================================
# Utility Functions
# ============================================================================
def _close_edges(
        edges: np.ndarray,
        kernel_size: tuple
    ) -> np.ndarray:
    """
    Purpose:
        Bridge along-line gaps in a Canny edge map so a fragmented lane line
        traces as one contour instead of several
        @NOTE detections have been broken and fragmented

    Inputs:
        edges: Canny edge image
        kernel_size: (width, height) of the morphological kernel

    Outputs:
        edges: Canny edge image
    """
    if not kernel_size or kernel_size[0] < 2:
        return edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

def _clamp(
        value: float, 
        lo: float, 
        hi: float
    ) -> float:
    """
    Purpose:
        Clamp value to range [lo, hi]

    Inputs:
        value: value to clamp
        lo: lower bound
        hi: upper bound

    Outputs:
        clamped value
    """
    return max(lo, min(hi, value))

def _canny(
        gray: np.ndarray, 
        params: CannyParams
    ) -> np.ndarray:
    """
    Purpose:
        Apply Canny edge detection to an assumed grayscale image with the given parameters

    Inputs:
        gray: grayscaled image
        params : CannyParams configuration

    Outputs:
        edges: Canny edge image
    """
    return cv2.Canny(gray, params.threshold1, params.threshold2,
                        apertureSize=params.aperture_size)

def _contours(
        edges: np.ndarray
    ):
    """
    Purpose:
        Extract external contours from a Canny edge image

    Inputs:
        edges: Canny edge image

    Outputs:
        contours: detected contours
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def _extreme_points(
        contour: np.ndarray
    ) -> tuple:
    """
    Purpose:
        Gives each contour a leftmost and rightmost point
        @NOTE used to combine possible fragmented contours

    Inputs:
        contour: detected contour

    Outputs:
        left : [x, y]
        right : [x, y]
    """
    pts = contour.reshape(-1, 2)
    return pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()]

# ============================================================================
# Internal Functions
# ============================================================================
def _mean_contour_intensity(
        gray: np.ndarray, 
        contour: np.ndarray
    ) -> float:
    """
    Purpose:
        Computes the mean pixel intensity within a contour, used to reject dark blobs such as seams
        @NOTE operation is separate from _extract_lane_candidates to isolate during testing
        @TODO integrate with _extract_lane_candidates operation

    Inputs:
        gray: grayscaled image
        contour: detected contour

    Outputs:
        mean_intensity: mean pixel intensity within the contour
    """
    x, y, w, h = cv2.boundingRect(contour)

    # Clamp to image bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, gray.shape[1]), min(y + h, gray.shape[0])

    # Reject contours that are too small
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Compute mean intensity
    roi_patch = gray[y1:y2, x1:x2]
    mask = np.zeros(roi_patch.shape, dtype=np.uint8)
    shifted = contour - np.array([[[x1, y1]]])
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    pixels = roi_patch[mask == 255]
    return float(np.mean(pixels)) if len(pixels) > 0 else 0.0

def _merge_collinear(
        candidates: List[LaneCandidate],
        lane_filter: LaneContourFilter,
        roi_shape: tuple,
        max_gap_px: float = 40.0,
    ) -> List[LaneCandidate]:
    """
    Purpose:
        Join fragments of one lane line into a single candidate so lane_offset
        cannot select two pieces of the same line as opposite boundaries

    Inputs:
        candidates: Accepted lane candidates
        lane_filter: LaneContourFilter configuration
        roi_shape: the shape of the roi image (roi_h, roi_w --- 2 dimensional)
        max_gap_px: The maximum distance between two points to be considered contiguous

    Outputs:
        candidates : Appended to the original list
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

        # csv output
        merged.append(LaneCandidate(
            label = "lane_boundary",
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
            proximity = round(_clamp((y + h) / max(roi_h, 1), 0.0, 1.0), 4),
            mean_intensity = mean_intensity,
            foot_x = contour_foot_x(pts.reshape(-1, 1, 2)),
        ))

    return merged + other

# ============================================================================
# Lane Boundary Detection
# ============================================================================
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
    Purpose:
        Compute a confidence score for a lane candidate based on the measurement quality. Ranks contours
        by how much their geometry can be trusted and should be treated as a weight

    Inputs:
        long: minAreaRect long dimension in px --- the long side of the candidate's bounding box
        short: minAreaRect short dimension in px --- the short side of the candidate's bounding box
        horizontal: True if a contours lines runs across the ROI horizontally
        mean_intensity: mean pixel intensity within the candidate's bounding box
        roi_h: lane ROI height
        roi_w: lane ROI width
        f: LaneContourFilter

    Outputs:
        confidence: clamped score between 0.0 and 1.0
    """

    # Normalized against the ROI span in the direction of the contour
    extent = roi_w if horizontal else roi_h
    ref_len = max(f.ref_length * max(extent, 1), 1.0)
    length_score = _clamp(long / ref_len, 0.0, 1.0)

    # Thickness separates tape from hairline edge traces
    width_score = _clamp(short / max(f.ref_width, 1.0), 0.0, 1.0)

    # Intensity is ranked based on how far above the threshold it is
    denom = max(255.0 - f.min_intensity, 1.0)
    intensity_score = _clamp((mean_intensity - f.min_intensity) / denom, 0.0, 1.0)

    score = 0.5 * length_score + 0.3 * intensity_score + 0.2 * width_score
    return round(_clamp(score, 0.0, 1.0), 4)

def _extract_lane_candidates(
    contours,
    lane_filter: LaneContourFilter,
    frame_id: int,
    timestamp_ms: int,
    roi_shape: tuple,
    gray: np.ndarray,
    reject_counts: dict | None = None   # @TODO move debug output into a dedicated logging version
) -> List[LaneCandidate]:
    """ Private Interface
    Purpose:
        Extracts contours that meet the input criteria
        @NOTE handles the filtering of detected contours
        @TODO remove debug outputs before sending to production (move to a dedicated debug operation)

    Inputs:
        contours: detected contours
        lane_filter: LaneFilter configuration parameters
        frame_id
        timestamp_ms
        roi_shape: The shaoe of the roi image (roi_h, roi_w --- 2 dimensional)
        gray: Grayscaled roi image --- used to compare intensities v. detected contours

        reject_counts: dict

    Outputs:
        candidates : list[LaneCandidate]
    """
    candidates = []
    roi_h, roi_w = roi_shape

    # Reject counters @TODO move debug output into a dedicated logging version
    rc = reject_counts if reject_counts is not None else {}
    for _k in ("seen", "area", "degenerate", "too_few_pts",
                "aspect", "w_span", "h_span", "intensity", "accepted"):
        rc.setdefault(_k, 0)

    # =============================
    # Contour Filtering Loop
    # =============================
    for contour in contours:
        rc["seen"] += 1     # Total number of seen contours

        # ===============
        # Contour Area
        area = cv2.contourArea(contour)
        if area < lane_filter.min_area or area > lane_filter.max_area:
            rc["area"] += 1
            continue

        # ===============
        # Bounding Rect
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            rc["degenerate"] += 1    # Reject degenerate contours
            continue
        if len(contour) < 5:
            rc["too_few_pts"] += 1   # Reject contours with too few points
            continue

        # ===============
        # Elongation
        _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        long_side = max(rect_w, rect_h)
        raw_short = min(rect_w, rect_h)
        short_side = max(raw_short, 1.0)
        elongation = long_side / short_side
        if elongation < lane_filter.min_aspect or elongation > lane_filter.max_aspect:
            rc["aspect"] += 1
            continue

        # ===============
        # Span
        horizontal = w >= h
        if horizontal and (w / roi_w) > lane_filter.max_roi_span:
            rc["w_span"] += 1   # Reject contours that are too wide
            continue
        if not horizontal and (h / roi_h) > lane_filter.max_roi_span:
            rc["h_span"] += 1   # Reject contours that are too tall
            continue

        # ===============
        # Intensity
        mean_intensity = _mean_contour_intensity(gray, contour)
        if mean_intensity < lane_filter.min_intensity:
            rc["intensity"] += 1
            continue

        # ===============
        # Confidence
        confidence = _lane_confidence(
            long = long_side,
            short = raw_short,
            horizontal = horizontal,
            mean_intensity = mean_intensity,
            roi_h = roi_h,
            roi_w = roi_w,
            f = lane_filter
        )

        # ===============
        # Proximity
        proximity = _clamp((y + h) / max(roi_h, 1), 0.0, 1.0)

        rc["accepted"] += 1 # Total number of accepted contours
        candidates.append(LaneCandidate(
            label = "lane_boundary",
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
) -> tuple:
    """ Public Interface
    Purpose:
        Extracts lane candidates from lane ROI using grayscale-Canny-contour pipeline and lane contour filter
        @NOTE handles the final candidate merging of data
        @TODO remove debug outputs before sending to production (move to a dedicated debug operation)

    Inputs:
        lane_roi: np.ndarray
        canny_params: CannyParams
        lane_filter: LaneContourFilter
        frame_id: int
        timestamp_ms: int

    Outputs:
        candidates : list[LaneCandidate]
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

    # Merges fragmented detections into a single contour shape
    candidates = _merge_collinear(candidates, lane_filter, lane_roi.shape[:2])  # 40 max_gap
    reject_counts["merged_into"] = len(candidates)

    # =========================================================
    # Debugging:    @TODO: Move to a dedicated debug operation
    # =========================================================
    debug_images = {
        "lane_roi": lane_roi,
        "edges": edges,
        "edges_raw": edges_raw,
        "reject_counts": reject_counts,
    } 
    if draw_overlays:
        # Return images to color for debugging
        contour_overlay = cv2.cvtColor(lane_roi, cv2.COLOR_GRAY2BGR)
        accepted_overlay = cv2.cvtColor(lane_roi, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(   # Draw all contours
            contour_overlay,
            contours,
            -1,
            (200, 200, 200),
            1
        )

        # Move through the candidates list
        for c in candidates:
            cv2.drawContours(   # Draw accepted contours
                accepted_overlay, 
                [c.contour], 
                -1, 
                (0, 255, 0), 
                2
            )
            x, y, w, h = c.bbox
            cv2.rectangle(  # Draw bounding box
                accepted_overlay, 
                (x, y), 
                (x + w - 1, y + h - 1), 
                (0, 200, 0), 
                2
            )
            cv2.putText(    # Draw confidence and proximity
                accepted_overlay, 
                f"{c.confidence:.2f}/{c.proximity:.2f}", 
                (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                cv2.LINE_AA
            )

        # Add images to debug dict
        debug_images["contour_overlay"] = contour_overlay
        debug_images["accepted_overlay"] = accepted_overlay

    # @TODO remove debug_images
    return candidates, debug_images

# ==============================================================================
# Sign Shape Extraction
# ==============================================================================
def _sign_confidence(
        area: float, 
        vertex_count: int, 
        f: SignContourFilter
    ) -> float:
    """
    Purpose: 
        Composite confidence score based on area and vertex count
    """
    vertex_score = _clamp(1.0 - abs(vertex_count - 8) / 8.0, 0.0, 1.0)
    denom = max(f.ref_area - f.min_area, 1.0)
    area_score = _clamp((area - f.min_area) / denom, 0.0, 1.0)
    return round(0.5 * vertex_score + 0.5 * area_score, 4)


# An area-rejected contour is traced only if its bounding box covers at least
# this fraction of min_area; smaller ones are edge noise and are only counted.
# Judged by bbox, not contour area: a Canny outline with a gap traces as a thin
# sliver with almost no enclosed area but a sign-sized box, and that is exactly
# the failure the trace is there to show
TRACE_MIN_BBOX_FRAC = 1.0

def _trace_entry(
        contour: np.ndarray,
        gate,
        area: float,
        vertices=None,
        solidity=None,
        confidence=None,
        poly=None,
    ) -> dict:
    """
    Purpose:
        One row of the sign trace: a contour the detector looked at and what
        it decided. Fields not measured before the contour was rejected are None

    Inputs:
        contour: the raw contour, used only for its bounding box
        gate: None if accepted, else "area", "vertices", "hull" or "solidity"
        area, vertices, solidity, confidence: measurements so far
        poly: the approxPolyDP polygon

    Outputs:
        dict with bbox, gate, area, vertices, solidity, confidence, poly
    """
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
    contours,
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
    reject_counts: dict | None = None,
    trace: list | None = None,
) -> List[SignCandidate]:
    """
    Purpose:
        Extract stop sign candidates from contours based on vertex count, area, and solidity

    Inputs:
        contours: detected contours
        sign_filter: SignContourFilter configuration
        frame_id
        timestamp_ms
        reject_counts: dict, filled with a count per gate
        trace: list, or None to skip. Given a list, one _trace_entry() is
               appended per contour that reached a gate, accepted or not, so
               a view can show what the detector saw and why it decided
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

        # Polygon approximation
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
            label = "stop_sign",
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
) -> tuple:
    """ Public interface
    Purpose:
        Extracts sign candidates from sign ROI using grayscale-Canny-contour pipeline and sign contour filter

    Inputs:
        sign_roi : np.ndarray
            Shape : (H_sign, W_sign)
            Dtype : uint8
            Color : Grayscale

        canny_params : CannyParams
            Threshold1, threshold2, and aperture size for cv2.Canny

        sign_filter : SignContourFilter
            Area, vertex count, and solidity thresholds

        draw_overlays : bool
            Build the contour and accepted overlays. False skips two full-ROI
            allocations and the contour rasterization per frame; the overlay
            keys are then absent from debug_images

        trace : bool
            Record every contour that reached a gate, with the gate that
            rejected it, under debug_images["trace"]. Off by default: the
            live loop does not read it

    Outputs:
        candidates : List[SignCandidate]
        debug_images : dict
            Always: sign_roi, edges, reject_counts.
            With trace: trace, a list of dicts (bbox, gate, area, vertices,
            solidity, confidence, poly), ROI-relative like every coordinate
            here. gate is None for the candidates that were accepted.
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
        # Both overlays must be 3-channel. Annotations are drawn with BGR
        # colors, and a single-channel destination takes only the first
        # component, which rendered every accepted contour, box and label
        # as black.
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

# =============================================================================
# Geometry Branch
# =============================================================================
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
) -> tuple:
    """
    Purpose:
        Runs the geometry branch on the given lane and sign ROIs with the specified parameters, returning
        the detected lane and sign candidates along with debug images.

    Inputs:
        lane_roi: uint8 grayscaled from ROICropResult.lane_roi
        sign_roi: uint8 grayscaled from ROICropResult.sign_roi
        canny_params: CannyParams configuration
        lane_filter: LaneContourFilter configuration
        sign_filters: SignContourFilter configuration
        frame_id
        timestamp_ms
        draw_overlays: build the debug overlays in both branches. False skips
                       four full-ROI allocations and two contour
                       rasterizations per frame
        trace: record every sign contour and the gate that decided it (see
               extract_sign_candidates)

    Outputs:
        result: GeometryBranchResult
        lane_debug: dict of debug images for lane ROI
        sign_debug: dict of debug images for sign ROI
    """
    # Input validation
    for name, roi in [("lane_roi", lane_roi), ("sign_roi", sign_roi)]:
        if roi is None:
            raise ValueError(   # Ensure the ROI is not empty
                f"run_geometry_branch: {name} is None"
            )
        if roi.dtype != np.uint8:
            raise TypeError(    # Ensure the ROI is the correct shape
                f"run_geometry_branch: {name} expected uint8, got {roi.dtype}"
            )
        if roi.ndim != 2:
            raise ValueError(   # Ensure the ROI is a single-channel grayscale image
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

# ============================================================================
# Geometry Branch Stage
# ============================================================================
def run_geometry_stage(
        roi: ROICropResult,
        config: GeometryConfig = GeometryConfig(),
        draw_overlays: bool = False,
        trace: bool = False,
    ) -> tuple:
    """
    Purpose:
        Stage entry point. Takes the previous stage's result and its config,
        the same shape as preprocess_frame() and crop_rois(), so the
        orchestrator chains single-argument calls instead of unpacking and
        re-threading the stamp by hand.

        This is a thin adapter over run_geometry_branch(). The detection
        path is unchanged and run_geometry_branch() remains callable
        directly with loose arrays for tuning work.

    Inputs:
        roi: ROICropResult from crop_rois()
        config: GeometryConfig tuning
        draw_overlays: default False. The live loop discards the debug dicts,
                       so building them costs allocations nothing reads.
                       The dataset harness passes True
        trace: default False. Records the per-contour sign trace for the
               debug views; see extract_sign_candidates

    Outputs:
        result: GeometryBranchResult
        lane_debug: dict of debug images for lane ROI
        sign_debug: dict of debug images for sign ROI

    Notes:
        frame_id and timestamp_ms come from the ROICropResult, which carried
        them from PreprocessResult, which carried them from FrameData. No
        stage re-derives either value.
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
