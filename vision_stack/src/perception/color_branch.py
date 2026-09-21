"""
color_branch.py

Color Branch Stage

Purpose:
    The color branch isolates traffic-light state candidates by color before any
    spatial or structural analysis occurs:

        traffic ROI (BGR) -> HSV -> red / yellow / green masks
                          -> blobs -> geometric filters -> candidates

    Returns an empty list if no candidates survive blob filtering.
    Coordinates are relative to the traffic-light ROI.

Where it runs:
    phase2_linker.run_chain() calls run_color_stage() after the geometry stage
    and passes the candidates to feature_fusion.fuse_detections(), which merges
    them with the geometry branch's lane and sign candidates. This module does
    not import fusion or the linker.

Calibration:
    The branch needs HSV ranges calibrated under the real course lighting,
    loaded with load_hsv_ranges(). ColorConfig() with no ranges leaves the
    branch OFF: run_color_stage() returns no candidates and says so in its
    debug dict. The HSVRanges() defaults are a scaffold, not a calibration.

Confidence:
    Area only: (blob area - min_area) / (ref_area - min_area), clamped to
    [0, 1]. It saturates at ref_area, and fusion keeps the highest-confidence
    traffic light across all three colors, so among several blobs the largest
    wins regardless of color or shape.

Debug:
    extract_traffic_light_candidates() always returns the masks, blob counts
    and mask pixel counts. With trace=True it also records every blob that
    reached a gate, and the gate that decided it, for the debug views.
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

# =============================================================================
# Configuration Dataclasses
# =============================================================================
DEFAULT_HSV_PATH = "vision_stack/calibration/hsv_ranges.json"

@dataclass
class ColorRange:
    """One HSV color band: lower and upper bounds as (H, S, V) uint8 tuples."""
    lower: tuple   # (H_min, S_min, V_min)
    upper: tuple   # (H_max, S_max, V_max)

@dataclass
class HSVRanges:
    """
    HSV thresholds for traffic light color detection

    Load calibrated values from calibration/hsv_ranges.json using
    load_hsv_ranges(). The defaults below are a structural scaffold only

    Hue is in OpenCV units, 0 to 179 (degrees / 2). Red needs two bands
    because hue wraps: red_low covers the bottom of the range, red_high the
    top
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
            True if load_hsv_ranges() was used. Informational: extraction does
            not refuse uncalibrated ranges, so they can be used while tuning,
            and the debug view flags them
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

@dataclass(frozen=True)
class ColorConfig:
    """
    The color branch's tuning as one unit, so the stage takes a single config
    argument like every other stage

    hsv_ranges: calibrated HSVRanges, or None to leave the branch off
    blob: BlobFilter
    """
    hsv_ranges: Optional[HSVRanges] = None
    blob: BlobFilter = field(default_factory=BlobFilter)

# =============================================================================
# Output Dataclasses
# =============================================================================
@dataclass
class TrafficLightCandidate:
    label: str    # "red" | "yellow" | "green"
    bbox: tuple  # (x, y, w, h) in traffic-light ROI coords
    confidence: float  # [0.0, 1.0] by area-based heuristic
    frame_id: int
    timestamp_ms: int

# =============================================================================
# Calibration Loader
# =============================================================================
_HSV_CAPS = (180, 255, 255)     # H allows 180 as an upper bound for the red wrap

def _band(data: dict, key: str) -> ColorRange:
    """One band from the JSON, checked so a typo fails at load, not on frame 1."""
    lower = tuple(int(v) for v in data[key]["lower"])
    upper = tuple(int(v) for v in data[key]["upper"])
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError(f"{key}: lower and upper each need 3 values (H, S, V)")
    for ch, lo, hi, cap in zip("HSV", lower, upper, _HSV_CAPS):
        if not (0 <= lo <= hi <= cap):
            raise ValueError(
                f"{key}: {ch} range [{lo}, {hi}] must satisfy "
                f"0 <= lower <= upper <= {cap}"
            )
    return ColorRange(lower, upper)

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
    Raises ValueError if a band is malformed: not 3 values, or a channel
    outside 0 <= lower <= upper <= 255 (hue: 180), naming the band and channel
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ranges = HSVRanges(
        red_low = _band(data, "red_low"),
        red_high = _band(data, "red_high"),
        yellow = _band(data, "yellow"),
        green = _band(data, "green"),
    )
    ranges._validated = True
    return ranges

def load_color_config(
        json_path: str = DEFAULT_HSV_PATH,
        blob: Optional[BlobFilter] = None,
    ) -> ColorConfig:
    """
    Purpose:
        ColorConfig with calibrated ranges from json_path, ready to switch the
        branch on in a PipelineConfig
    """
    return ColorConfig(load_hsv_ranges(json_path), blob or BlobFilter())

# =============================================================================
# Utility Functions
# =============================================================================
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

def _mean_hsv(
        hsv: np.ndarray,
        contour: np.ndarray
    ) -> tuple:
    """
    Purpose:
        Mean (H, S, V) inside a contour, to show where a blob sits relative to
        the calibrated bands. Trace mode only

    Inputs:
        hsv: HSV image of the ROI
        contour: the blob's contour
    """
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([[[x, y]]])], -1, 255,
                     thickness=cv2.FILLED)
    m = cv2.mean(hsv[y:y + h, x:x + w], mask=mask)
    return (round(m[0], 1), round(m[1], 1), round(m[2], 1))

# An area-rejected blob is traced only if its bounding box covers at least this
# fraction of min_area; smaller ones are mask speckle and are only counted
TRACE_MIN_BBOX_FRAC = 1.0

def _blobs_to_candidates(
    mask: np.ndarray,
    label: str,
    blob_filter: BlobFilter,
    frame_id: int,
    timestamp_ms: int,
    reject_counts: Optional[dict] = None,
    trace: Optional[list] = None,
    hsv: Optional[np.ndarray] = None,
) -> List[TrafficLightCandidate]:
    """
    Purpose:
        Extract blobs from a binary mask, apply geometric filters, and return
        surviving blobs as TrafficLightCandidate objects

    Inputs:
        mask: binary mask for one color
        label: "red" | "yellow" | "green"
        reject_counts: dict, filled with seen / area / aspect / accepted
        trace: list, or None to skip. Given a list, one entry per blob that
               reached a gate is appended: label, bbox, gate (None if
               accepted, else "area" or "aspect"), area, aspect, fill,
               confidence, hsv
        hsv: HSV image, used only to fill the trace's mean HSV
    """
    candidates = []

    rc = reject_counts if reject_counts is not None else {}
    for _k in ("seen", "area", "aspect", "accepted"):
        rc.setdefault(_k, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def note(contour, gate, area, bbox, aspect=None, confidence=None):
        if trace is None:
            return
        x, y, w, h = bbox
        trace.append({
            "label": label,
            "bbox": bbox,
            "gate": gate,
            "area": area,
            "aspect": None if aspect is None else round(aspect, 3),
            "fill": round(area / (w * h), 3) if w * h else None,
            "confidence": confidence,
            "hsv": _mean_hsv(hsv, contour) if hsv is not None else None,
        })

    for contour in contours:
        rc["seen"] += 1
        area = cv2.contourArea(contour)

        # Area filter
        if area < blob_filter.min_area or area > blob_filter.max_area:
            rc["area"] += 1
            if trace is not None:
                bbox = cv2.boundingRect(contour)
                if bbox[2] * bbox[3] >= blob_filter.min_area * TRACE_MIN_BBOX_FRAC:
                    note(contour, "area", area, bbox)
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Aspect ratio filter that guards against a zero height
        if h == 0:
            rc["aspect"] += 1
            continue
        aspect = w / h

        # Confidence is the normalized area relative to reference area
        confidence = round(_clamp(
            (area - blob_filter.min_area) / max(blob_filter.ref_area - blob_filter.min_area, 1.0),
            0.0, 1.0
        ), 4)

        if aspect < blob_filter.min_aspect or aspect > blob_filter.max_aspect:
            rc["aspect"] += 1
            note(contour, "aspect", area, (x, y, w, h), aspect, confidence)
            continue

        rc["accepted"] += 1
        note(contour, None, area, (x, y, w, h), aspect, confidence)
        candidates.append(TrafficLightCandidate(
            label = label,
            bbox = (x, y, w, h),
            confidence = confidence,
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
        ))

    return candidates

# =============================================================================
# Core Function
# =============================================================================
def extract_traffic_light_candidates(
    roi: np.ndarray,
    hsv_ranges: HSVRanges,
    blob_filter: BlobFilter,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    trace: bool = False,
) -> tuple:
    """
    Purpose:
        Extract traffic-light color candidates from the traffic-light ROI

    Inputs:
        roi: uint8 BGR ndarray (@TODO Change to YUV)
        hsv_ranges: HSVRanges, normally calibrated and loaded with
                    load_hsv_ranges(). Uncalibrated ranges are accepted so
                    they can be tuned; debug["calibrated"] reports which
        blob_filter: BlobFilter geometric constraints
        frame_id: integer frame counter from capture loop
        timestamp_ms: millisecond timestamp from capture loop
        trace: also record every blob that reached a gate, under
               debug["trace"]. Off by default: the live loop does not read it

    Outputs:
        (candidates, debug)

        candidates: list[TrafficLightCandidate]
        debug: dict
            Always: hsv, red, yellow, green (images), roi (the input),
            mask_px ({color: nonzero pixels}), reject_counts
            ({color: {seen, area, aspect, accepted}}), calibrated (bool)
            With trace: trace, a list of dicts (see _blobs_to_candidates)

    Note:
        debug is not a part of the final output in the pipeline
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
            "Load from calibration/hsv_ranges.json via load_hsv_ranges()."
        )

    # Step 1: BGR → HSV 
    hsv = _to_hsv(roi)

    # Step 2: Per-color thresholding -> binary masks
    masks = {
        "red": _threshold_red(hsv, hsv_ranges),
        "yellow": _threshold_single(hsv, hsv_ranges.yellow),
        "green": _threshold_single(hsv, hsv_ranges.green),
    }

    # Step 3: Blob filtering -> candidates
    candidates = []
    reject_counts = {}
    trace_log = [] if trace else None
    for label, mask in masks.items():
        rc = reject_counts[label] = {}
        candidates += _blobs_to_candidates(
            mask, label, blob_filter, frame_id, timestamp_ms,
            rc, trace_log, hsv if trace else None,
        )

    debug = {
        "hsv": hsv,
        "red": masks["red"],
        "yellow": masks["yellow"],
        "green": masks["green"],
        "roi": roi,
        "mask_px": {k: int(cv2.countNonZero(m)) for k, m in masks.items()},
        "reject_counts": reject_counts,
        "calibrated": hsv_ranges.is_calibrated,
    }
    if trace:
        debug["trace"] = trace_log

    return candidates, debug

# =============================================================================
# Color Branch Stage
# =============================================================================
def _traffic_roi_bgr(roi, frame_bgr: np.ndarray) -> np.ndarray:
    """
    Purpose:
        The traffic ROI in BGR. The color branch needs color, and the ROI the
        crop stage hands out may be grayscale (geometry's are), so fall back
        to cutting the original frame at the same rect

    Notes:
        The fallback is only right if the crop's rect is in the coordinates of
        the frame given here, i.e. preprocessing did not resize or warp it.
        The shapes are compared so a mismatch raises instead of analysing the
        wrong pixels
    """
    img = roi.traffic_roi
    if img is not None and img.ndim == 3 and img.shape[2] == 3:
        return img

    x, y, w, h = roi.traffic_rect
    crop = frame_bgr[y:y + h, x:x + w]
    if img is not None and crop.shape[:2] != img.shape[:2]:
        raise ValueError(
            f"run_color_stage: traffic_roi is {img.shape[:2]} but traffic_rect "
            f"{roi.traffic_rect} cut {crop.shape[:2]} from the frame. The crop "
            f"and the frame are not in the same coordinates (preprocess "
            f"resized or warped?), so the color ROI cannot be recovered from "
            f"the frame"
        )
    return crop

def run_color_stage(
        roi,
        frame_bgr: np.ndarray,
        config: ColorConfig = ColorConfig(),
        trace: bool = False,
    ) -> tuple:
    """
    Purpose:
        Stage entry point, the same shape as run_geometry_stage(): takes the
        crop stage's result and a config, and carries the frame stamp from the
        ROICropResult instead of re-deriving it

    Inputs:
        roi: ROICropResult from crop_rois()
        frame_bgr: the frame as capture delivered it, used only when the ROI
                   the crop stage hands out is not already color
        config: ColorConfig. With no hsv_ranges the branch is off
        trace: record the per-blob trace (see extract_traffic_light_candidates)

    Outputs:
        (candidates, debug). Off: ([], {"enabled": False}). On: the debug dict
        from extract_traffic_light_candidates plus "enabled": True
    """
    if config.hsv_ranges is None:
        return [], {"enabled": False}

    candidates, debug = extract_traffic_light_candidates(
        _traffic_roi_bgr(roi, frame_bgr),
        config.hsv_ranges,
        config.blob,
        roi.frame_id,
        roi.timestamp_ms,
        trace,
    )
    debug["enabled"] = True
    return candidates, debug

# =============================================================================
# Debug Visualization
# =============================================================================
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