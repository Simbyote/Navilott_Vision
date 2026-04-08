"""
geometry_branch.py
==================
Geometry Branch — Vision Perception Phase (Phase 2, Step 4b of 7)

Operates on the lane ROI and sign ROI produced by roi_crop.py.
Produces lane-boundary candidates and sign-shape candidates using grayscale
conversion, Canny edge detection, and contour geometry filtering only.
No color analysis, no neural inference, no temporal filtering.

Purpose
-------
The geometry branch answers two structural questions from the frame:

  1. Where are the lane boundaries?
     Lane markings are white tape on a dark mat — intensity discontinuities
     that Canny detects reliably. Contours are filtered by aspect ratio and
     area to accept long, thin, roughly horizontal or vertical shapes that
     match lane line geometry, and reject background clutter.

  2. Is there a stop sign shape present?
     The stop sign is an octagon. Contour approximation (cv2.approxPolyDP)
     reduces a contour to its dominant vertices. An octagon produces
     approximately 8 vertices. Area and convexity filtering further
     discriminate against noise contours. No color is used here — the
     geometry branch is colorblind by design.

These two detections operate on different ROIs and are logically separated
even though they share the grayscale→Canny→contour pipeline structure.
The results of both are returned together for handoff to feature fusion.

Input contract
--------------
  lane_roi : np.ndarray
      Shape  : (H_lane, W_lane, 3)
      Dtype  : uint8
      Color  : BGR
      Source : ROICropResult.lane_roi from roi_crop.py
               (preprocessed — undistorted, equalized, blurred)

  sign_roi : np.ndarray
      Shape  : (H_sign, W_sign, 3)
      Dtype  : uint8
      Color  : BGR
      Source : ROICropResult.sign_roi from roi_crop.py
               (preprocessed — undistorted, equalized, blurred)

  canny_params : CannyParams
      Threshold1, threshold2, and aperture size for cv2.Canny.
      MISSING DEPENDENCY — thresholds must be tuned per Calibration Step 4
      (edge threshold verification on actual course hardware and lighting).
      The defaults in CannyParams are starting points only.
      Applied to both lane and sign ROIs independently.

  lane_filter : LaneContourFilter
      Geometric bounds for accepting a contour as a lane-boundary candidate.
      Defaults are starting points. Tune against actual course lane tape width
      and expected range from camera.

  sign_filter : SignContourFilter
      Geometric bounds for accepting a contour as a sign-shape candidate.
      Defaults are starting points. Tune against actual stop sign dimensions
      and expected detection range.

  frame_id : int
      Frame counter from capture loop. Carried into each candidate.

  timestamp_ms : int
      Millisecond timestamp from capture loop. Carried into each candidate.

Output contract
---------------
  Returns : GeometryBranchResult

  .lane_candidates  list[LaneCandidate]
      Each element:
          .label        str   — "lane_boundary"
          .bbox         tuple — (x, y, w, h) in lane ROI pixel coords
          .contour      np.ndarray — raw contour points, shape (N, 1, 2)
          .confidence   float — [0.0, 1.0]; see confidence definition
          .frame_id     int
          .timestamp_ms int

  .sign_candidates  list[SignCandidate]
      Each element:
          .label        str   — "stop_sign"
          .bbox         tuple — (x, y, w, h) in sign ROI pixel coords
          .contour      np.ndarray — raw contour points, shape (N, 1, 2)
          .vertex_count int   — number of vertices from approxPolyDP
          .confidence   float — [0.0, 1.0]; see confidence definition
          .frame_id     int
          .timestamp_ms int

  All coordinates are ROI-relative. Feature fusion is responsible for
  re-projecting into source frame coordinates.

Confidence definitions
----------------------
  Lane:
      confidence = clamp(
          (contour_area - min_area) / (ref_area - min_area),
          0.0, 1.0
      )
      Area-based — larger contour = stronger lane marking signal.

  Sign:
      vertex_score = 1.0 - abs(vertex_count - 8) / 8.0
      area_score   = clamp((contour_area - min_area) / (ref_area - min_area), 0.0, 1.0)
      confidence   = 0.5 * vertex_score + 0.5 * area_score
      Vertex proximity to 8 is the primary discriminator for an octagon.
      Area provides a secondary size signal.

Failure cases
-------------
  F1 — lane_roi or sign_roi is None / wrong shape / wrong dtype:
       Asserted at entry. Raises ValueError or TypeError.

  F2 — Canny thresholds too low:
       Mask floods with edges — too many contours, all pass area filter,
       runtime spikes. Symptom: hundreds of candidates returned.
       Mitigation: raise threshold1 and threshold2 in CannyParams.

  F3 — Canny thresholds too high:
       Lane edges are not detected. Zero candidates returned.
       Mitigation: lower thresholds or verify histogram equalization is active.

  F4 — Lane ROI contains intersection (no parallel lines):
       Contour geometry does not match lane filter criteria. Few or zero
       candidates. This is expected behavior — Phase 3 handles missing lane
       detections.

  F5 — Sign ROI does not contain a stop sign:
       Zero sign candidates. Expected. Phase 3 handles absent detections.

  F6 — Sign approximation produces wrong vertex count due to noisy contour:
       approxPolyDP epsilon too small → too many vertices retained.
       Mitigation: increase epsilon_factor in SignContourFilter.

  F7 — frame_id or timestamp_ms both 0:
       Candidates are ambiguous in time. Capture loop must supply correct values.

  F8 — ROI is too small for min_area filter:
       All contours rejected. Returns empty candidate lists. Not an error.
"""

import os
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CannyParams:
    """
    Parameters for cv2.Canny edge detection.

    MISSING DEPENDENCY: threshold1 and threshold2 must be verified against
    actual course frames per Calibration Step 4. These defaults are
    structural starting points only.

    threshold1    : lower hysteresis threshold
    threshold2    : upper hysteresis threshold
    aperture_size : Sobel kernel size (3, 5, or 7 only)
    """
    threshold1:    float = 50.0
    threshold2:    float = 150.0
    aperture_size: int   = 3


@dataclass
class LaneContourFilter:
    """
    Geometric acceptance criteria for lane-boundary contours.

    min_area     : minimum bounding-box area (px²) — rejects dust and noise
    max_area     : maximum bounding-box area (px²) — rejects full-frame blobs
    min_aspect   : min(w/h, h/w) lower bound — lane lines are elongated
    ref_area     : area treated as confidence = 1.0 at expected detection range

    min_aspect logic: lane lines may be horizontal or vertical depending on
    heading. Use min(w/h, h/w) to test elongation regardless of orientation.
    A square blob has min(w/h, h/w) = 1.0 and is rejected by min_aspect < 1.0.
    """
    min_area:   float = 30.0
    max_area:   float = 20000.0
    min_aspect: float = 2.0    # elongation threshold: long side / short side ≥ 2
    ref_area:   float = 2000.0


@dataclass
class SignContourFilter:
    """
    Geometric acceptance criteria for sign-shape (octagon) contours.

    min_area       : minimum contour area (px²)
    max_area       : maximum contour area (px²)
    min_vertices   : approxPolyDP vertex count lower bound
    max_vertices   : approxPolyDP vertex count upper bound
                     8-gon should produce 6-10 depending on epsilon
    min_solidity   : contour_area / convex_hull_area — octagon is convex
    epsilon_factor : approxPolyDP epsilon = epsilon_factor * arc_length
                     smaller = more vertices retained; larger = fewer
    ref_area       : area treated as confidence area_score = 1.0
    """
    min_area:      float = 200.0
    max_area:      float = 30000.0
    min_vertices:  int   = 6
    max_vertices:  int   = 10
    min_solidity:  float = 0.80
    epsilon_factor: float = 0.03
    ref_area:      float = 5000.0


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Internal helpers — shared
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _to_grayscale(roi_bgr: np.ndarray) -> np.ndarray:
    """BGR → grayscale. No copy of input retained."""
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)


def _run_canny(gray: np.ndarray, params: CannyParams) -> np.ndarray:
    """Apply Canny edge detection. Returns binary edge image."""
    return cv2.Canny(gray, params.threshold1, params.threshold2,
                     apertureSize=params.aperture_size)


def _extract_contours(edges: np.ndarray):
    """Extract external contours from a Canny edge image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# ---------------------------------------------------------------------------
# Lane boundary extraction
# ---------------------------------------------------------------------------

def _lane_confidence(area: float, f: LaneContourFilter) -> float:
    denom = max(f.ref_area - f.min_area, 1.0)
    return _clamp((area - f.min_area) / denom, 0.0, 1.0)


def _extract_lane_candidates(
    contours,
    lane_filter: LaneContourFilter,
    frame_id: int,
    timestamp_ms: int,
) -> List[LaneCandidate]:
    """
    Filter contours to lane-boundary candidates.

    Elongation test: a lane line is long relative to its width.
    min(w/h, h/w) < 1/min_aspect rejects compact blobs regardless of
    which dimension is larger.
    """
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < lane_filter.min_area or area > lane_filter.max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            continue

        # Elongation: accept if longer dimension is at least min_aspect × shorter
        elongation = max(w, h) / max(min(w, h), 1)
        if elongation < lane_filter.min_aspect:
            continue

        confidence = _lane_confidence(area, lane_filter)

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
    Run grayscale → Canny → contour extraction → lane geometry filter
    on the lane ROI.

    Returns (candidates, debug_images)
      candidates   : list[LaneCandidate]
      debug_images : dict — 'gray', 'edges', 'contour_overlay', 'accepted_overlay'
    """
    gray     = _to_grayscale(lane_roi)
    edges    = _run_canny(gray, canny_params)
    contours = _extract_contours(edges)

    candidates = _extract_lane_candidates(contours, lane_filter, frame_id, timestamp_ms)

    # Debug overlays (computed only when harness requests them)
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


# ---------------------------------------------------------------------------
# Sign shape extraction
# ---------------------------------------------------------------------------

def _sign_confidence(area: float, vertex_count: int, f: SignContourFilter) -> float:
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
    Filter contours to stop-sign (octagon) candidates.

    Sequence:
      1. Area filter
      2. approxPolyDP vertex count filter (target: ~8 for octagon)
      3. Solidity filter (octagon is convex — high solidity)
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

        # Solidity: reject non-convex / fragmented shapes
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
            contour      = approx,   # approximated polygon, not raw contour
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
    Run grayscale → Canny → contour extraction → sign geometry filter
    on the sign ROI.

    Returns (candidates, debug_images)
      candidates   : list[SignCandidate]
      debug_images : dict — 'gray', 'edges', 'contour_overlay', 'accepted_overlay'
    """
    gray     = _to_grayscale(sign_roi)
    edges    = _run_canny(gray, canny_params)
    contours = _extract_contours(edges)

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


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

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
    Run the full geometry branch on both ROIs.

    Parameters
    ----------
    lane_roi     : uint8 BGR — from ROICropResult.lane_roi
    sign_roi     : uint8 BGR — from ROICropResult.sign_roi
    canny_params : CannyParams — shared for both ROIs
    lane_filter  : LaneContourFilter
    sign_filter  : SignContourFilter
    frame_id     : int from capture loop
    timestamp_ms : int from capture loop

    Returns
    -------
    (result, lane_debug, sign_debug)

    result      : GeometryBranchResult
    lane_debug  : dict of debug images for lane ROI
    sign_debug  : dict of debug images for sign ROI
    """
    # --- Guard: input validation -------------------------------------------
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


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Standalone test harness.

    Input priority (per sample dir):
      1. Pre-cropped ROI files from roi_crop.py:
           sN/results/*_roi_lane.png
           sN/results/*_roi_sign.png
      2. If not found: loads full images from sN/ and applies inline crop
         using the same coordinate definitions as roi_crop.py.

    Output per image → sN/results/:
      stem_gb_lane_gray.png          — lane ROI grayscale
      stem_gb_lane_edges.png         — lane Canny edges
      stem_gb_lane_contours.png      — all contours overlaid
      stem_gb_lane_accepted.png      — accepted lane candidates
      stem_gb_sign_gray.png          — sign ROI grayscale
      stem_gb_sign_edges.png         — sign Canny edges
      stem_gb_sign_contours.png      — all contours overlaid
      stem_gb_sign_accepted.png      — accepted sign candidates
    """
    import os

    SAMPLE_DIRS = [
        "vision_stack/frames/trackT3",
        "vision_stack/frames/trackT4",
        "vision_stack/frames/trackT5"
    ]
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    canny_params = CannyParams()       # tune per Calibration Step 4
    lane_filter  = LaneContourFilter() # tune against course hardware
    sign_filter  = SignContourFilter() # tune against course hardware

    total_ok   = 0
    total_fail = 0
    frame_id   = 0

    for sample_dir in SAMPLE_DIRS:
        if not os.path.isdir(sample_dir):
            print(f"[SKIP] Not found: {sample_dir}")
            continue

        results_dir = os.path.join(sample_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # --- Attempt to find pre-cropped lane and sign ROI pairs ------------
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

            # --- Write lane debug images ------------------------------------
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_lane_gray.png"),
                        lane_debug["gray"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_lane_edges.png"),
                        lane_debug["edges"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_lane_contours.png"),
                        lane_debug["contour_overlay"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_gb_lane_accepted.png"),
                        lane_debug["accepted_overlay"])

            # --- Write sign debug images ------------------------------------
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