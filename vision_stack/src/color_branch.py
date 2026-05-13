"""
color_branch.py

Color Branch Stage

Purpose:
    The color branch isolates traffic-light state candidates by color before any
    spatial or structural analysis occurs

Returns an empty list if no candidates survive blob filtering
Coordinates are relative to the traffic-light ROI
"""
import sys
import cv2
import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import List

# ==========================================================
# Dataset Loading Utility
# ==========================================================
sys.path.insert(0, "vision_stack/src")

from unzip_data import fetch_dataset

# ============================================================================
# Configuration Dataclasses
# ============================================================================
@dataclass
class ColorRange:
    """One HSV color band: lower and upper bounds as (H, S, V) uint8 tuples."""
    lower: tuple   # (H_min, S_min, V_min)
    upper: tuple   # (H_max, S_max, V_max)

@dataclass
class HSVRanges:
    """
    Calibrated HSV thresholds for traffic light color detection

    load from calibration/hsv_ranges.json using load_hsv_ranges()
    This placeholder is a structural scaffold only

    Red requires two bands because hue wraps at 180°
    red_low covers [0°, ~10°]
    red_high covers [~170°, 180°]
    """
    red_low: ColorRange = field(default_factory=lambda: ColorRange((0, 100, 100), (10, 255, 255)))
    red_high: ColorRange = field(default_factory=lambda: ColorRange((170, 100, 100), (180, 255, 255)))
    yellow: ColorRange = field(default_factory=lambda: ColorRange((20, 100, 100), (35, 255, 255)))
    green: ColorRange = field(default_factory=lambda: ColorRange((40, 100, 100), (80, 255, 255)))

    def __post_init__(self):
        """
        Purpose:
            Verifies load_hsv_ranges() was used
        """
        self._validated = False   # True only after load_hsv_ranges() is used

    @property
    def is_calibrated(self) -> bool:
        """
        Purpose:
            True if load_hsv_ranges() was used
        """
        return self._validated

@dataclass
class BlobFilter:
    """
    Geometric constraints for accepting a blob as a traffic-light candidate

    min_area: discard blobs smaller than this. Rejects noise
    max_area: discard blobs larger than this. Rejects large
                    background regions mistakenly masked
    min_aspect: w/h lower bound. Rejects elongated streaks
    max_aspect: w/h upper bound. Rejects elongated streaks
    ref_area: blob area treated as confidence = 1.0 at expected
                    detection range; used in confidence normalization

    Starting defaults below are placeholders
    """
    min_area: float = 50.0
    max_area: float = 5000.0
    min_aspect: float = 0.3
    max_aspect: float = 3.0
    ref_area: float = 800.0

# ============================================================================
# Output Dataclasses
# ============================================================================
@dataclass
class TrafficLightCandidate:
    label: str    # "red" | "yellow" | "green"
    bbox: tuple  # (x, y, w, h) in traffic-light ROI coords
    confidence: float  # [0.0, 1.0] by area-based heuristic
    frame_id: int
    timestamp_ms: int

# ============================================================================
# Calibration Loader
# ============================================================================
def load_hsv_ranges(json_path: str) -> HSVRanges:
    """
    Load calibrated HSV thresholds from calibration/hsv_ranges.json

    JSON structure:
    {
      "red_low": {"lower": [H, S, V], "upper": [H, S, V]},
      "red_high": {"lower": [H, S, V], "upper": [H, S, V]},
      "yellow": {"lower": [H, S, V], "upper": [H, S, V]},
      "green": {"lower": [H, S, V], "upper": [H, S, V]}
    }

    Raises FileNotFoundError if the file does not exist
    Raises KeyError if any required key is missing
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ranges = HSVRanges(
        red_low = ColorRange(tuple(data["red_low"]["lower"]), tuple(data["red_low"]["upper"])),
        red_high = ColorRange(tuple(data["red_high"]["lower"]), tuple(data["red_high"]["upper"])),
        yellow = ColorRange(tuple(data["yellow"]["lower"]), tuple(data["yellow"]["upper"])),
        green = ColorRange(tuple(data["green"]["lower"]), tuple(data["green"]["upper"])),
    )
    ranges._validated = True
    return ranges


# ============================================================================
# Utility Functions
# ============================================================================
def _to_hsv(
        roi_bgr: np.ndarray
    ) -> np.ndarray:
    """
    Purpose:
        Convert BGR ROI to HSV

    Note:
        @TODO Does not convert from a YUV image; it assumes BGR
    """
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

def _threshold_red(
        hsv: np.ndarray, 
        ranges: HSVRanges
    ) -> np.ndarray:
    """
    Purpose:
        Produce red binary mask
    """
    mask_low = cv2.inRange(hsv,
                            np.array(ranges.red_low.lower, dtype=np.uint8),
                            np.array(ranges.red_low.upper, dtype=np.uint8))
    mask_high = cv2.inRange(hsv,
                            np.array(ranges.red_high.lower, dtype=np.uint8),
                            np.array(ranges.red_high.upper, dtype=np.uint8))
    return cv2.bitwise_or(mask_low, mask_high)

def _threshold_single(
        hsv: np.ndarray, 
        color_range: ColorRange
    ) -> np.ndarray:
    """
    Purpose:
        Produce binary mask for a single non-wrapping hue range
    """
    return cv2.inRange(hsv,
                       np.array(color_range.lower, dtype=np.uint8),
                       np.array(color_range.upper, dtype=np.uint8))


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


def _blobs_to_candidates(
    mask: np.ndarray,
    label: str,
    blob_filter: BlobFilter,
    frame_id: int,
    timestamp_ms: int,
) -> List[TrafficLightCandidate]:
    """
    Purpose:
        Extract connected components from a binary mask, apply geometric filters,
        and return surviving blobs as TrafficLightCandidate objects
    """
    candidates = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Area filter
        if area < blob_filter.min_area or area > blob_filter.max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Aspect ratio filter that guards against a zero height
        if h == 0:
            continue
        aspect = w / h
        if aspect < blob_filter.min_aspect or aspect > blob_filter.max_aspect:
            continue

        # Confidence is the normalized area relative to reference area
        confidence = _clamp(
            (area - blob_filter.min_area) / max(blob_filter.ref_area - blob_filter.min_area, 1.0),
            0.0, 1.0
        )

        candidates.append(TrafficLightCandidate(
            label = label,
            bbox = (x, y, w, h),
            confidence = round(confidence, 4),
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
        ))

    return candidates

# ============================================================================
# Core Function
# ============================================================================
def extract_traffic_light_candidates(
    roi: np.ndarray,
    hsv_ranges: HSVRanges,
    blob_filter: BlobFilter,
    frame_id: int = 0,
    timestamp_ms: int = 0,
) -> tuple:
    """
    Purpose:
        Extract traffic-light color candidates from the traffic-light ROI

    Inputs:
        roi: uint8 BGR ndarray (@TODO Change to YUV)
        hsv_ranges: calibrated HSVRanges to be loaded via load_hsv_ranges()
        blob_filter: BlobFilter geometric constraints
        frame_id: integer frame counter from capture loop
        timestamp_ms: millisecond timestamp from capture loop

    Outputs:
        (candidates, debug_masks)

        candidates: list[TrafficLightCandidate]
        debug_masks: dict with keys 'hsv', 'red', 'yellow', 'green'
                    Values are np.ndarray images for debug output
    
    Note:
        debug_mask is not a part of the final output in the pipeline
    """
    # Guards and input validation
    if roi is None:
        raise ValueError("extract_traffic_light_candidates: received None ROI")
    if roi.dtype != np.uint8:
        raise TypeError(f"extract_traffic_light_candidates: expected uint8, got {roi.dtype}")
    if roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError(f"extract_traffic_light_candidates: expected (H,W,3) BGR, got {roi.shape}")
    if hsv_ranges is None:
        raise ValueError(
            "extract_traffic_light_candidates: hsv_ranges is required. "
            "Load from calibration/hsv_ranges.json via load_hsv_ranges(). "
            "Do not pass None or use unvalidated defaults in production."
        )

    # Step 1: BGR → HSV (@TODO Change to YUV)
    hsv = _to_hsv(roi)

    # Step 2: Per-color thresholding -> binary masks
    mask_red = _threshold_red(hsv, hsv_ranges)
    mask_yellow = _threshold_single(hsv, hsv_ranges.yellow)
    mask_green = _threshold_single(hsv, hsv_ranges.green)

    # Step 3: Blob filtering -> candidates
    candidates = []
    candidates += _blobs_to_candidates(mask_red, "red", blob_filter, frame_id, timestamp_ms)
    candidates += _blobs_to_candidates(mask_yellow, "yellow", blob_filter, frame_id, timestamp_ms)
    candidates += _blobs_to_candidates(mask_green, "green", blob_filter, frame_id, timestamp_ms)

    debug_masks = {
        "hsv": hsv,
        "red": mask_red,
        "yellow": mask_yellow,
        "green": mask_green,
    }

    return candidates, debug_masks

# ============================================================================
# Debug Visualization
# ============================================================================
_LABEL_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 200, 255),
    "green": (0, 200, 0),
}

def draw_candidates(
        roi_bgr: np.ndarray, 
        candidates: List[TrafficLightCandidate]
    ) -> np.ndarray:
    """
    Purpose:
        Return a copy of roi_bgr with candidate bounding boxes and labels drawn
    """
    vis = roi_bgr.copy()
    for c in candidates:
        x, y, w, h = c.bbox
        color = _LABEL_COLORS.get(c.label, (255, 255, 255))
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.putText(vis, f"{c.label} {c.confidence:.2f}",
                    (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis

# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    """
    Standalone Test (traffic-light candidates extraction)

    Purpose:
        Run the color branch on a set of sample images from the course

    Inputs: 
        Traffic-light ROI images from sN/results/ (written by roi_crop.py)
        Specifically looks for files matching *_roi_traffic.png

    If those do not exist, falls back to loading the full image from sN/ and
    applying the traffic-light ROI crop inline (same coordinates as roi_crop.py)

    Output per image -> vision_stack/frames/trackT*/results/:
      stem_cb_hsv.png: HSV representation (H channel only, false-color)
      stem_cb_mask_red.png: red binary mask
      stem_cb_mask_yellow.png: yellow binary mask
      stem_cb_mask_green.png: green binary mask
      stem_cb_candidates.png: ROI with candidate bounding boxes

    Notes:
        If calibration/hsv_ranges.json is absent, the test prints an explicit error and exits
    """
    import os
    import sys

    SAMPLE_DIRS = fetch_dataset(
        url="https://github.com/Simbyote/Navilott_Vision/releases/download/v1.0-dataset/frame_tracks.zip",
        zip_path="vision_stack/frames/frame_tracks.zip",
        dest_dir="vision_stack/frames",
    )
    HSV_RANGES_PATH = "vision_stack/calibration/hsv_ranges.json"  # for real detections
    HSV_DUMMY_PATH = "vision_stack/dummy/dummy_hsv_ranges.json"  # for unit test scaffolding
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    # Load calibrated HSV ranges, 
    if not os.path.exists(HSV_RANGES_PATH):
        print(
            f"[ERROR] {HSV_RANGES_PATH} not found.\n"
            "HSV range calibration is required before running this substage.\n"
            "Run the interactive trackbar tuning tool under actual course lighting\n"
            "and save the result to calibration/hsv_ranges.json."
        )
        try:
            hsv_ranges = load_hsv_ranges(HSV_DUMMY_PATH)
            print(f"[WARNING] Using dummy HSV ranges from {HSV_DUMMY_PATH} are not valid for real detection!")
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to load dummy HSV ranges from {HSV_DUMMY_PATH}: {e}")
            sys.exit(1)
    else:
        try:
            hsv_ranges = load_hsv_ranges(HSV_RANGES_PATH)
            print(f"[INFO] HSV ranges loaded from {HSV_RANGES_PATH}")
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to parse {HSV_RANGES_PATH}: {e}")
            sys.exit(1)

    blob_filter = BlobFilter()   # placeholder defaults

    total_ok = 0
    total_fail = 0
    frame_id = 0

    for sample_dir in SAMPLE_DIRS:
        if not os.path.isdir(sample_dir):
            print(f"[SKIP] Not found: {sample_dir}")
            continue

        results_dir = os.path.join(sample_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Prefer pre-cropped traffic ROIs from roi_crop.py output
        roi_files = sorted(
            f for f in os.listdir(results_dir)
            if f.endswith("_roi_traffic.png")
        ) if os.path.isdir(results_dir) else []

        # Fall back to full images with inline crop
        if not roi_files:
            source_files = sorted(
                f for f in os.listdir(sample_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            )
            use_inline_crop = True
        else:
            source_files = roi_files
            use_inline_crop = False

        if not source_files:
            print(f"[SKIP] No input images in {sample_dir}")
            continue

        for filename in source_files:
            src_path = os.path.join(
                results_dir if not use_inline_crop else sample_dir,
                filename
            )
            stem = filename.replace("_roi_traffic", "")
            stem = os.path.splitext(stem)[0]

            img = cv2.imread(src_path)
            if img is None:
                print(f"[FAIL] Could not read: {src_path}")
                total_fail += 1
                continue

            # Inline crop if pre-cropped ROI not available
            if use_inline_crop:
                H, W = img.shape[:2]
                tl_x = W // 4
                tl_w = W - 2 * (W // 4)
                tl_h = H // 2
                roi = img[0:tl_h, tl_x:tl_x + tl_w]
            else:
                roi = img

            ts_ms = int(time.time() * 1000)

            try:
                candidates, debug_masks = extract_traffic_light_candidates(
                    roi = roi,
                    hsv_ranges = hsv_ranges,
                    blob_filter = blob_filter,
                    frame_id = frame_id,
                    timestamp_ms = ts_ms,
                )
            except (ValueError, TypeError) as e:
                print(f"[FAIL] {src_path}: {e}")
                total_fail += 1
                continue

            # Debug image 1: HSV (H channel only, false-color) =================
            # Visualize only the H channel rescaled to [0,255]
            # gives a readable hue map
            # ==================================================================
            h_channel = debug_masks["hsv"][:, :, 0]
            hsv_vis = cv2.applyColorMap(h_channel, cv2.COLORMAP_HSV)
            cv2.imwrite(os.path.join(results_dir, f"{stem}_cb_hsv.png"), hsv_vis)

            # Debug images 2–4: binary masks ===================================
            # Generate the individual binary masks for each color (red, yellow, green)
            # ================================================================== 
            cv2.imwrite(os.path.join(results_dir, f"{stem}_cb_mask_red.png"), debug_masks["red"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_cb_mask_yellow.png"), debug_masks["yellow"])
            cv2.imwrite(os.path.join(results_dir, f"{stem}_cb_mask_green.png"), debug_masks["green"])

            # Debug image 5: candidate visualization ===========================
            # Determine the best candidates and draw them on the image
            # ==================================================================
            vis = draw_candidates(roi, candidates)
            cv2.imwrite(os.path.join(results_dir, f"{stem}_cb_candidates.png"), vis)

            label_summary = [f"{c.label}({c.confidence:.2f})" for c in candidates]
            print(
                f"[OK] frame_id={frame_id}  {src_path}"
                f"candidates={label_summary if label_summary else '[]'}"
            )
            frame_id += 1
            total_ok += 1

    print(f"\nDone. {total_ok} processed, {total_fail} failed.")