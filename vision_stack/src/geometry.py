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

All coordinates are ROI-relative. Feature fusion is responsible for
re-projecting into source frame coordinates.
"""
import os
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List


# ============================================================================
# Input Dataclasses
# ============================================================================

@dataclass
class CannyParams:  # Edge Detection
    """
    Parameters for cv2.Canny edge detection

    threshold1: lower hysteresis threshold
    threshold2: upper hysteresis threshold
    aperture_size: Sobel kernel size (3, 5, or 7)
    """
    threshold1:    float = 10.0
    threshold2:    float = 150.0
    aperture_size: int   = 3

@dataclass
class LaneContourFilter:    # Lane Boundary
    """
    Geometric acceptance criteria for lane-boundary contours

    min_area: minimum bounding-box area; rejects dust and noise
    max_area: maximum bounding-box area; rejects full-frame blobs
    min_aspect: min(w/h, h/w) lower bound; lane lines are elongated
    max_aspect: min(w/h, h/w) upper bound; rejects extreme aspect ratios
    ref_area: area treated as confidence = 1.0 at expected detection range
    max_roi_span: fraction of ROI a contour may span in its elongated axis
    min_intensity: minimum average intensity (0-255) within contour; rejects dark blobs
    min_aspect logic: lane lines may be horizontal or vertical depending on heading
    """
    min_area:   float = 0.0
    max_area:   float = 300.0
    min_aspect: float = 8.0
    max_aspect:   float = 10.0
    ref_area:   float = 2000.0
    max_roi_span: float = 1.0
    min_intensity: float = 80.0

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
    min_area:      float = 200.0
    max_area:      float = 30000.0
    min_vertices:  int   = 6
    max_vertices:  int   = 10
    min_solidity:  float = 0.80
    epsilon_factor: float = 0.03
    ref_area:      float = 5000.0

# ============================================================================
# Output Dataclasses
# ===========================================================================

@dataclass
class LaneCandidate:
    label:        str          # "lane_boundary"
    bbox:         tuple        # (x, y, w, h) in lane ROI coords
    contour:      np.ndarray   # shape (N, 1, 2), int32
    confidence:   float        # [0.0, 1.0]
    frame_id:     int
    timestamp_ms: int

@dataclass
class SignCandidate:
    label:        str          # "stop_sign"
    bbox:         tuple        # (x, y, w, h) in sign ROI coords
    contour:      np.ndarray   # shape (N, 1, 2), int32
    vertex_count: int          # from approxPolyDP
    confidence:   float        # [0.0, 1.0]
    frame_id:     int
    timestamp_ms: int

@dataclass
class GeometryBranchResult:
    lane_candidates: List[LaneCandidate]
    sign_candidates: List[SignCandidate]
    frame_id:        int
    timestamp_ms:    int

# ============================================================================
# Utility Functions
# ============================================================================

def _clamp(
        value: float, 
        lo: float, 
        hi: float
    ) -> float:
    """
    Purpose:
        Clamp value to range [lo, hi]
    """
    return max(lo, min(hi, value))

def _to_grayscale(
        roi: np.ndarray
    ) -> np.ndarray:
    """
    Purpose:
        Convert YUV image to grayscale
    """
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_YUV2BGR)
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

def _canny(
        gray: np.ndarray, 
        params: CannyParams
    ) -> np.ndarray:
    """
    Purpose:
        Apply Canny edge detection to a grayscale image with the given parameters
    """
    return cv2.Canny(gray, params.threshold1, params.threshold2,
                     apertureSize=params.aperture_size)

def _contours(
        edges: np.ndarray
    ):
    """
    Purpose:
        Extract external contours from a Canny edge image
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# ============================================================================
# Lane Boundary Detection
# ============================================================================

def _lane_confidence(
        area: float, 
        elongation: float, 
        bbox: tuple,
        roi_h: int, f: LaneContourFilter
    ) -> float:
    """
    Purpose:
        Computes the confidence score of a detected lane boundary contour based on its 
        geometric properties

    Inputs:
        area         : contour area
        elongation   : contour aspect ratio
        bbox         : (x, y, w, h) in lane ROI coords
        roi_h        : lane ROI height
        f            : LaneContourFilter

    Outputs:
        confidence   : [0.0, 1.0]
    """
    # How far above the minimum elongation threshold? A stronger lane shape = higher score
    elong_score = _clamp(
        (elongation - f.min_aspect) / max(f.max_aspect - f.min_aspect, 1.0),
        0.0, 1.0
    )
    # Size signal, normalized against expected detection area
    area_score = _clamp(area / max(f.ref_area, 1.0), 0.0, 1.0)

    # Vertical proximity, contours near the bottom of the lane ROI are closer to camera
    x, y, w, h = bbox
    proximity_score = _clamp((y + h) / max(roi_h, 1), 0.0, 1.0)

    return round(0.50 * elong_score + 0.30 * area_score + 0.20 * proximity_score, 4)


def _mean_contour_intensity(
        gray: np.ndarray, 
        contour: np.ndarray
    ) -> float:
    """
    Purpose:
        Computes the mean pixel intensity within a contour, used to reject dark blobs such as seams

    Inputs:
        gray    : grayscale image
        contour : shape (N, 1, 2), int32

    Outputs:
        mean_intensity : [0.0, 255.0]
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

def _extract_lane_candidates(
    contours,
    lane_filter: LaneContourFilter,
    frame_id: int,
    timestamp_ms: int,
    roi_shape: tuple,
    gray: np.ndarray,
) -> List[LaneCandidate]:
    """
    Purpose:
        Extracts contours that meet the input criteria

    Inputs:
        contours     : list[np.ndarray]
        lane_filter  : LaneContourFilter
        frame_id     : int
        timestamp_ms : int
        roi_shape    : tuple
        gray         : np.ndarray

    Outputs:
        candidates : list[LaneCandidate]
    """
    candidates = []
    roi_h, roi_w = roi_shape

    for contour in contours:
        area = cv2.contourArea(contour)

        # Reject contours that are too small
        if area < lane_filter.min_area or area > lane_filter.max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            continue
        if len(contour) < 5:
            continue

        # Accept if longer dimension is at least min_aspect x shorter
        _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        long_side  = max(rect_w, rect_h)
        short_side = max(min(rect_w, rect_h), 1.0)
        elongation = long_side / short_side

        # Reject contours that are too elongated
        if elongation < lane_filter.min_aspect:
            continue

        # Reject contours that span more than max_roi_span of ROI
        horizontal = rect_w >= rect_h
        if horizontal and (w / roi_w) > lane_filter.max_roi_span:
            continue
        if not horizontal and (h / roi_h) > lane_filter.max_roi_span:
            continue

        # Reject dark contours, such as seams
        mean_intensity = _mean_contour_intensity(gray, contour)
        if mean_intensity < lane_filter.min_intensity:
            continue

        # Composite confidence score based on area and elongation
        confidence = _lane_confidence(area, elongation, (x, y, w, h), roi_h, lane_filter)

        candidates.append(LaneCandidate(
            label        = "lane_boundary",
            bbox         = (x, y, w, h),
            contour      = contour,
            confidence   = round(confidence, 4),
            frame_id     = frame_id,
            timestamp_ms = timestamp_ms,
        ))

    return candidates

def extract_lane_candidates(
    lane_roi: np.ndarray,
    canny_params: CannyParams,
    lane_filter: LaneContourFilter,
    frame_id: int,
    timestamp_ms: int,
) -> tuple:
    """
    Purpose:
        Extracts lane candidates from lane ROI using grayscale-Canny-contour pipeline and lane contour filter

    Inputs:
        lane_roi     : np.ndarray
        canny_params : CannyParams
        lane_filter  : LaneContourFilter
        frame_id     : int
        timestamp_ms : int

    Outputs:
        candidates : list[LaneCandidate]
    """
    gray     = _to_grayscale(lane_roi)
    edges    = _canny(gray, canny_params)
    contours = _contours(edges)

    candidates = _extract_lane_candidates(
        contours, lane_filter, frame_id, timestamp_ms,
        roi_shape=lane_roi.shape[:2], gray = gray
    )
    # Debug overlays
    contour_overlay  = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    accepted_overlay = lane_roi.copy()

    cv2.drawContours(contour_overlay, contours, -1, (200, 200, 200), 1)
    for c in candidates:
        cv2.drawContours(accepted_overlay, [c.contour], -1, (0, 255, 0), 2)
        x, y, w, h = c.bbox
        cv2.rectangle(accepted_overlay, (x, y), (x + w - 1, y + h - 1), (0, 200, 0), 1)
        cv2.putText(accepted_overlay, f"{c.confidence:.2f}",
                    (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

    debug_images = {
        "gray":             gray,
        "edges":            edges,
        "contour_overlay":  contour_overlay,
        "accepted_overlay": accepted_overlay,
    }

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
    denom        = max(f.ref_area - f.min_area, 1.0)
    area_score   = _clamp((area - f.min_area) / denom, 0.0, 1.0)
    return round(0.5 * vertex_score + 0.5 * area_score, 4)


def _extract_sign_candidates(
    contours,
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
) -> List[SignCandidate]:
    """
    Purpose:
        Extract stop sign candidates from contours based on vertex count, area, and solidity
    """
    candidates = []


    for contour in contours:
        area = cv2.contourArea(contour)
        if area < sign_filter.min_area or area > sign_filter.max_area:
            continue

        # Polygon approximation
        arc_len  = cv2.arcLength(contour, closed=True)
        epsilon  = sign_filter.epsilon_factor * arc_len
        approx   = cv2.approxPolyDP(contour, epsilon, closed=True)
        n_verts  = len(approx)

        if n_verts < sign_filter.min_vertices or n_verts > sign_filter.max_vertices:
            continue

        # Reject non-convex / fragmented shapes
        hull     = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < sign_filter.min_solidity:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        confidence  = _sign_confidence(area, n_verts, sign_filter)

        candidates.append(SignCandidate(
            label        = "stop_sign",
            bbox         = (x, y, w, h),
            contour      = approx,
            vertex_count = n_verts,
            confidence   = confidence,
            frame_id     = frame_id,
            timestamp_ms = timestamp_ms,
        ))

    return candidates

def extract_sign_candidates(
    sign_roi: np.ndarray,
    canny_params: CannyParams,
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
) -> tuple:
    """
    Purpose:
        Extracts sign candidates from sign ROI using grayscale-Canny-contour pipeline and sign contour filter

    Inputs:
        sign_roi : np.ndarray
            Shape  : (H_sign, W_sign, 3)
            Dtype  : uint8
            Color  : YUV

        canny_params : CannyParams
            Threshold1, threshold2, and aperture size for cv2.Canny

        sign_filter : SignContourFilter
            Area, vertex count, and solidity thresholds

    Outputs:
        candidates : List[SignCandidate]
    """
    gray     = _to_grayscale(sign_roi)
    edges    = _canny(gray, canny_params)
    contours = _contours(edges)

    candidates = _extract_sign_candidates(contours, sign_filter, frame_id, timestamp_ms)

    # Debug overlays
    contour_overlay  = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    accepted_overlay = sign_roi.copy()

    cv2.drawContours(contour_overlay, contours, -1, (200, 200, 200), 1)
    for c in candidates:
        cv2.drawContours(accepted_overlay, [c.contour], -1, (0, 0, 255), 2)
        x, y, w, h = c.bbox
        cv2.rectangle(accepted_overlay, (x, y), (x + w - 1, y + h - 1), (0, 0, 200), 1)
        cv2.putText(accepted_overlay, f"v={c.vertex_count} {c.confidence:.2f}",
                    (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

    debug_images = {
        "gray":             gray,
        "edges":            edges,
        "contour_overlay":  contour_overlay,
        "accepted_overlay": accepted_overlay,
    }

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
) -> tuple:
    """
    Purpose:
        Runs the geometry branch on the given lane and sign ROIs with the specified parameters, returning
        the detected lane and sign candidates along with debug images.

    Inputs:
        lane_roi: uint8 BGR — from ROICropResult.lane_roi
        sign_roi: uint8 BGR — from ROICropResult.sign_roi
        canny_params: CannyParams — shared for both ROIs
        lane_filter: LaneContourFilter
        sign_filters: SignContourFilter
        frame_id: int from capture loop
        timestamp_ms: int from capture loop

    Outputs:
        result: GeometryBranchResult
        lane_debug: dict of debug images for lane ROI
        sign_debug: dict of debug images for sign ROI
    """
    # Input validation
    for name, roi in [("lane_roi", lane_roi), ("sign_roi", sign_roi)]:
        if roi is None:
            raise ValueError(f"run_geometry_branch: {name} is None")
        if roi.dtype != np.uint8:
            raise TypeError(f"run_geometry_branch: {name} expected uint8, got {roi.dtype}")
        if roi.ndim != 3 or roi.shape[2] != 3:
            raise ValueError(f"run_geometry_branch: {name} expected (H,W,3) BGR, got {roi.shape}")

    lane_candidates, lane_debug = extract_lane_candidates(
        lane_roi, canny_params, lane_filter, frame_id, timestamp_ms)

    sign_candidates, sign_debug = extract_sign_candidates(
        sign_roi, canny_params, sign_filter, frame_id, timestamp_ms)

    result = GeometryBranchResult(
        lane_candidates = lane_candidates,
        sign_candidates = sign_candidates,
        frame_id        = frame_id,
        timestamp_ms    = timestamp_ms,
    )

    return result, lane_debug, sign_debug

# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    """
    Standalone test (static dataset):

    Purpose:
        Run the geometry branch on a set of sample images from the course, producing debug outputs at
        each step

    Outputs:
      stem_gb_lane_gray.png: gray-scale lane ROI
      stem_gb_lane_edges.png: Canny edges from lane ROI
      stem_gb_lane_contours.png: lane ROI with all contours drawn
      stem_gb_lane_accepted.png: lane ROI with accepted contours drawn
      stem_gb_sign_gray.png: gray-scale sign ROI
      stem_gb_sign_edges.png: Canny edges from sign ROI
      stem_gb_sign_contours.png: sign ROI with all contours drawn
      stem_gb_sign_accepted.png: sign ROI with accepted contours drawn
    """
    import os

    SAMPLE_DIRS = [
        "vision_stack/frames/trackT3",
        "vision_stack/frames/trackT4",
        "vision_stack/frames/trackT5"
    ]
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    canny_params = CannyParams()
    lane_filter  = LaneContourFilter()
    sign_filter  = SignContourFilter()

    total_ok   = 0
    total_fail = 0
    frame_id   = 0

    for sample_dir in SAMPLE_DIRS:
        if not os.path.isdir(sample_dir):
            print(f"[SKIP] Not found: {sample_dir}")
            continue

        results_dir = os.path.join(sample_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Attempt to find pre-cropped lane and sign ROI pairs
        roi_lane_files = {}
        roi_sign_files = {}

        if os.path.isdir(results_dir):
            for f in os.listdir(results_dir):
                if f.endswith("_roi_lane.png"):
                    stem = f.replace("_roi_lane.png", "")
                    roi_lane_files[stem] = os.path.join(results_dir, f)
                elif f.endswith("_roi_sign.png"):
                    stem = f.replace("_roi_sign.png", "")
                    roi_sign_files[stem] = os.path.join(results_dir, f)

        # Stems present in both lane and sign sets
        paired_stems = sorted(set(roi_lane_files) & set(roi_sign_files))

        # Fall back to full images if no paired ROIs found
        use_inline_crop = len(paired_stems) == 0
        full_images = []
        if use_inline_crop:
            if use_inline_crop:
                full_images = sorted(
                    f for f in os.listdir(sample_dir)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                )
            if not full_images:
                print(f"[SKIP] No inputs in {sample_dir}")
                continue
            
        stems_to_process = paired_stems if not use_inline_crop else [
            os.path.splitext(f)[0] for f in full_images
        ]

        for stem in stems_to_process:
            ts_ms = int(time.time() * 1000)

            if use_inline_crop:
                img_path = os.path.join(sample_dir, f"{stem}{next((os.path.splitext(f)[1] for f in full_images if os.path.splitext(f)[0] == stem), '.png')}")
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[FAIL] Could not read: {img_path}")
                    total_fail += 1
                    continue
                H, W = img.shape[:2]
                # lane ROI: lower half
                lane_roi = img[H // 2 : H,  0 : W]
                # sign ROI: right half, full height
                sign_roi = img[0 : H,  W // 2 : W]
            else:
                lane_img = cv2.imread(roi_lane_files[stem])
                sign_img = cv2.imread(roi_sign_files[stem])
                if lane_img is None or sign_img is None:
                    print(f"[FAIL] Could not read ROI pair for stem: {stem}")
                    total_fail += 1
                    continue
                lane_roi = lane_img
                sign_roi = sign_img

            try:
                result, lane_debug, sign_debug = run_geometry_branch(
                    lane_roi     = lane_roi,
                    sign_roi     = sign_roi,
                    canny_params = canny_params,
                    lane_filter  = lane_filter,
                    sign_filter  = sign_filter,
                    frame_id     = frame_id,
                    timestamp_ms = ts_ms,
                )
            except (ValueError, TypeError) as e:
                print(f"[FAIL] {stem}: {e}")
                total_fail += 1
                continue

            # debug lane images
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_lane_gray.png"),
                        lane_debug["gray"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_lane_edges.png"),
                        lane_debug["edges"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_lane_contours.png"),
                        lane_debug["contour_overlay"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_lane_accepted.png"),
                        lane_debug["accepted_overlay"])

            # debug sign images
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_sign_gray.png"),
                        sign_debug["gray"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_sign_edges.png"),
                        sign_debug["edges"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_sign_contours.png"),
                        sign_debug["contour_overlay"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_sign_accepted.png"),
                        sign_debug["accepted_overlay"])

            lane_summary = f"{len(result.lane_candidates)} lane"
            sign_summary = (
                [f"v={c.vertex_count} conf={c.confidence:.2f}" for c in result.sign_candidates]
                or "[]"
            )
            print(
                f"[OK] frame_id={frame_id}  {stem}  "
                f"{lane_summary} candidates | sign={sign_summary}"
            )
            frame_id += 1
            total_ok += 1

    print(f"\nDone. {total_ok} processed, {total_fail} failed.")