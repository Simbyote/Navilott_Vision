"""
geometry.py

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
re-projecting into source frame coordinates
"""
import sys
import os
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List

# ==========================================================
# Dataset Loading Utility
# ==========================================================
sys.path.insert(0, "vision_stack/src")

from unzip_data import fetch_dataset

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
    threshold1: float = 10.0
    threshold2: float = 300.0
    aperture_size: int = 3

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
    min_area: float = 1.0        # was 0.0 -> "area < 0.0" could never fire
    max_area: float = 200.0
    min_aspect: float = 5.0
    max_aspect: float = 300.0      # NOW ENFORCED as a rejection (was unused)
    ref_area: float = 300.0       # was 2000.0, above max_area -> capped score
    max_roi_span: float = 0.60    # was 1.0 -> "ratio > 1.0" could never fire
    min_intensity: float = 20.0
    edge_margin_frac: float = 0.08   # lateral prior; see _lane_confidence
    edge_penalty: float = 0.35       # multiplier for contours in the margin

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
    """
    label: str
    bbox: tuple
    contour: np.ndarray
    confidence: float
    frame_id: int
    timestamp_ms: int

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
    """
    label: str
    bbox: tuple
    contour: np.ndarray
    vertex_count: int
    confidence: float
    frame_id: int
    timestamp_ms: int

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

# Set True ONLY together with re-calibrating CannyParams and
# LaneContourFilter.min_intensity. See the note in _to_grayscale().
INPUT_IS_BGR = True

def _to_grayscale(
        roi: np.ndarray
    ) -> np.ndarray:
    """
    Purpose:
        Convert the ROI to grayscale.

    WARNING -- do not flip INPUT_IS_BGR on its own.
        run_pipeline.py passes BGR into this branch, but the legacy path below
        runs COLOR_YUV2BGR on it, reinterpreting BGR data as YUV and pushing it
        through the YUV->BGR matrix. The resulting "grayscale" is a scrambled
        linear combination of the true channels.

        Every downstream constant -- CannyParams(10, 160),
        LaneContourFilter.min_intensity=80, and _mean_contour_intensity's
        behaviour -- was tuned empirically AGAINST that scrambled image. They
        are therefore self-consistent and currently working. Correcting the
        conversion without re-calibrating those three constants in the same
        commit will make lane detection worse, not better.

        Flipping this flag also removes one full-ROI cvtColor per branch per
        frame, which is worth having given 11.2% of frames exceeded the 66.7 ms
        budget in session2.log.
    """
    if INPUT_IS_BGR:
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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
        roi_h: int, roi_w: int, f: LaneContourFilter
    ) -> float:
    """
    Purpose:
        Computes the confidence score of a detected lane boundary contour based on its 
        geometric properties

    Inputs:
        area: contour area
        elongation: contour aspect ratio
        bbox: (x, y, w, h) in lane ROI coords
        roi_h: lane ROI height
        roi_w: lane ROI width
        f: LaneContourFilter

    Outputs:
        confidence: [0.0, 1.0]
    """
    # Elongation is BAND-PASS, not monotonic. The old form rewarded elongation
    # without bound and saturated at 1.0 for anything past max_aspect, so a
    # continuous wall/floor seam (elongation ~25) scored higher than a real
    # dashed lane marking (elongation ~9). A lane dash has a BOUNDED aspect
    # ratio; only scenery is arbitrarily long. Score peaks mid-band.
    band_lo, band_hi = f.min_aspect, f.max_aspect
    band_mid = 0.5 * (band_lo + band_hi)
    band_half = max(0.5 * (band_hi - band_lo), 1.0)
    elong_score = _clamp(1.0 - abs(elongation - band_mid) / band_half, 0.0, 1.0)

    # Size signal. ref_area is now <= max_area, so this can actually reach 1.0.
    # Previously ref_area=2000 with max_area=300 capped area_score at 0.15,
    # meaning its 0.30 weight contributed at most 0.045 and lane confidence
    # could never exceed 0.745.
    area_score = _clamp(area / max(min(f.ref_area, f.max_area), 1.0), 0.0, 1.0)

    # Vertical proximity: contours near the bottom of the lane ROI are closer
    # to the camera and more trustworthy.
    x, y, w, h = bbox
    proximity_score = _clamp((y + h) / max(roi_h, 1), 0.0, 1.0)

    score = 0.45 * elong_score + 0.30 * area_score + 0.25 * proximity_score

    # Lateral prior. A contour hugging the left or right margin of the lane ROI
    # is far more likely to be a wall base, table leg, or mat edge than a lane
    # marking -- the robot cannot be so far out of lane that a boundary reaches
    # the frame edge while the pipeline still claims a valid lane. This is the
    # geometry-side half of the wall fix; the estimator-side half is the width
    # band in lane_offset.compute_lane_offset().
    if roi_w > 0:
        margin = f.edge_margin_frac * roi_w
        if x < margin or (x + w) > (roi_w - margin):
            score *= f.edge_penalty

    return round(_clamp(score, 0.0, 1.0), 4)

def _mean_contour_intensity(
        gray: np.ndarray, 
        contour: np.ndarray
    ) -> float:
    """
    Purpose:
        Computes the mean pixel intensity within a contour, used to reject dark blobs such as seams

    Inputs:
        gray: grayscale image
        contour: shape (N, 1, 2), int32

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
        contours: list[np.ndarray]
        lane_filter: LaneContourFilter
        frame_id: int
        timestamp_ms : int
        roi_shape: tuple
        gray: np.ndarray

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
        long_side = max(rect_w, rect_h)
        short_side = max(min(rect_w, rect_h), 1.0)
        elongation = long_side / short_side

        # Reject contours outside the lane-shape aspect band.
        # max_aspect was documented as "upper bound; rejects extreme aspect
        # ratios" but had no rejection branch -- it was only an EMA normaliser.
        # This is what let an unbounded wall seam through.
        if elongation < lane_filter.min_aspect or elongation > lane_filter.max_aspect:
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
        confidence = _lane_confidence(area, elongation, (x, y, w, h), roi_h, roi_w, lane_filter)

        candidates.append(LaneCandidate(
            label = "lane_boundary",
            bbox = (x, y, w, h),
            contour = contour,
            confidence = round(confidence, 4),
            frame_id = frame_id,
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
        lane_roi: np.ndarray
        canny_params: CannyParams
        lane_filter: LaneContourFilter
        frame_id: int
        timestamp_ms: int

    Outputs:
        candidates : list[LaneCandidate]
    """
    gray = _to_grayscale(lane_roi)
    edges = _canny(gray, canny_params)
    contours = _contours(edges)

    candidates = _extract_lane_candidates(
        contours, lane_filter, frame_id, timestamp_ms,
        roi_shape=lane_roi.shape[:2], gray = gray
    )
    # Debug overlays
    contour_overlay = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
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
        "gray": gray,
        "edges": edges,
        "contour_overlay": contour_overlay,
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
    denom = max(f.ref_area - f.min_area, 1.0)
    area_score = _clamp((area - f.min_area) / denom, 0.0, 1.0)
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
        confidence = _sign_confidence(area, n_verts, sign_filter)

        candidates.append(SignCandidate(
            label = "stop_sign",
            bbox = (x, y, w, h),
            contour = approx,
            vertex_count = n_verts,
            confidence = confidence,
            frame_id = frame_id,
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
            Shape : (H_sign, W_sign, 3)
            Dtype : uint8
            Color : YUV

        canny_params : CannyParams
            Threshold1, threshold2, and aperture size for cv2.Canny

        sign_filter : SignContourFilter
            Area, vertex count, and solidity thresholds

    Outputs:
        candidates : List[SignCandidate]
    """
    gray = _to_grayscale(sign_roi)
    edges = _canny(gray, canny_params)
    contours = _contours(edges)

    candidates = _extract_sign_candidates(contours, sign_filter, frame_id, timestamp_ms)

    # Debug overlays
    contour_overlay = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
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
        "gray": gray,
        "edges": edges,
        "contour_overlay": contour_overlay,
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
        lane_roi: uint8 BGR from ROICropResult.lane_roi (@TODO: change to YUV)
        sign_roi: uint8 BGR from ROICropResult.sign_roi (@TODO: change to YUV)
        canny_params: CannyParams shared for both ROIs
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
        frame_id = frame_id,
        timestamp_ms = timestamp_ms,
    )

    return result, lane_debug, sign_debug

# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    """
    Standalone test (geometry extraction):

    Purpose:
        Run the geometry branch on a set of sample images from the course, producing debug outputs at
        each step

    Outputs per image -> vision_stack/frames/trackT*/results/:
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
        "vision_stack/frames/Sample1",
        "vision_stack/frames/Sample2",
        "vision_stack/frames/Sample3"
    ]

    '''
    SAMPLE_DIRS = fetch_dataset(
        url="https://github.com/Simbyote/Navilott_Vision/releases/download/v1.0-dataset/frame_tracks.zip",
        zip_path="vision_stack/frames/frame_tracks.zip",
        dest_dir="vision_stack/frames",
    )
    '''
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    canny_params = CannyParams()
    lane_filter = LaneContourFilter()
    sign_filter = SignContourFilter()

    total_ok = 0
    total_fail = 0
    frame_id = 0

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
                print(f"[SKIP] no ROI fixtures in {results_dir} — run roi_crop.py first")
                continue
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
                    lane_roi = lane_roi,
                    sign_roi = sign_roi,
                    canny_params = canny_params,
                    lane_filter = lane_filter,
                    sign_filter = sign_filter,
                    frame_id = frame_id,
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