"""Color branch: traffic-light color candidates from the traffic ROI.

Purpose:
    Isolates traffic-light state by color before any structural analysis.
    Runs after the geometry stage in phase2_linker.run_chain(); its
    candidates go to feature_fusion.fuse_detections() alongside the lane and
    sign candidates. The branch stays off until HSV ranges calibrated under
    course lighting are loaded; the built-in ranges are only a scaffold.

Main package:
    TrafficLightCandidate: a color label, a traffic-ROI-relative bbox and an
    area-based confidence for one blob. An empty list means no blob survived.

Flow:
    1. Convert the BGR traffic ROI to HSV.
    2. Threshold into red, yellow and green masks (red spans the hue wrap).
    3. Find blobs in each mask and gate them by area and aspect.
    4. Score survivors by area and package them with the frame identity.
"""
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.params import GREEN, HSV_RANGES_PATH, RED, YELLOW
from src.utils import clamp
from src.perception.roi_crop import ROICropResult

DEFAULT_HSV_PATH = str(HSV_RANGES_PATH)

@dataclass
class ColorRange:
    """One HSV color band, bounds inclusive."""
    lower: tuple[int, int, int]   # (H_min, S_min, V_min)
    upper: tuple[int, int, int]   # (H_max, S_max, V_max)

@dataclass
class HSVRanges:
    """
    HSV thresholds per light color.

    The defaults are a structural scaffold; load calibrated values with
    load_hsv_ranges(). Hue is in OpenCV units, 0-179 (degrees / 2). Red needs
    two bands because hue wraps: red_low covers the bottom of the range,
    red_high the top.
    """
    red_low: ColorRange = field(default_factory=lambda: ColorRange((0, 100, 100), (10, 255, 255)))
    red_high: ColorRange = field(default_factory=lambda: ColorRange((170, 100, 100), (180, 255, 255)))
    yellow: ColorRange = field(default_factory=lambda: ColorRange((20, 100, 100), (35, 255, 255)))
    green: ColorRange = field(default_factory=lambda: ColorRange((40, 100, 100), (80, 255, 255)))

    def __post_init__(self):
        self._validated = False   # True only after load_hsv_ranges() is used

    @property
    def is_calibrated(self) -> bool:
        """
        True only for ranges built by load_hsv_ranges(). Informational:
        extraction still accepts scaffold ranges for tuning, and the debug
        dict flags them.
        """
        return self._validated

@dataclass
class BlobFilter:
    """Blob gates for traffic-light candidates. Defaults are placeholders, not tuned."""
    min_area: float = 50.0      # px^2; rejects mask speckle
    max_area: float = 5000.0    # px^2; rejects large background regions caught by a band
    min_aspect: float = 0.3     # w/h; together with max_aspect, rejects elongated streaks
    max_aspect: float = 3.0
    ref_area: float = 800.0     # px^2 scoring confidence 1.0: the expected lamp size at detection range

@dataclass(frozen=True)
class ColorConfig:
    """Color tuning as one unit, so the stage takes a single config like every other stage."""
    hsv_ranges: HSVRanges | None = None     # None leaves the branch off
    blob: BlobFilter = field(default_factory=BlobFilter)


@dataclass
class TrafficLightCandidate:
    """One color blob that passed the gates. frame_id and timestamp_ms are carried from capture."""
    label: str                          # "red" | "yellow" | "green"
    bbox: tuple[int, int, int, int]     # (x, y, w, h) in traffic-ROI px
    # [0, 1], area only, saturating at ref_area. Fusion keeps the highest
    # confidence across all three colors, so the largest blob wins regardless
    # of color or shape.
    confidence: float
    frame_id: int
    timestamp_ms: int


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
    Load calibrated HSV thresholds and mark them calibrated.

    Inputs:
        json_path: JSON with a lower/upper [H, S, V] pair per band:
            {
              "red_low": {"lower": [H, S, V], "upper": [H, S, V]},
              "red_high": {"lower": [H, S, V], "upper": [H, S, V]},
              "yellow": {"lower": [H, S, V], "upper": [H, S, V]},
              "green": {"lower": [H, S, V], "upper": [H, S, V]}
            }

    Outputs:
        HSVRanges with is_calibrated True.

    Side effects:
        Reads json_path.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        KeyError: If a band or bound is missing.
        ValueError: If a bound doesn't have 3 values, or a channel falls
            outside 0 <= lower <= upper <= 255 (hue: 180). The message names
            the band and channel.
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
        blob: BlobFilter | None = None,
    ) -> ColorConfig:
    """
    ColorConfig with calibrated ranges, ready to switch the branch on in a PipelineConfig.

    Inputs:
        json_path: Calibration file; see load_hsv_ranges().
        blob: Blob gates. None uses the BlobFilter() placeholders.
    """
    return ColorConfig(load_hsv_ranges(json_path), blob or BlobFilter())


# @TODO assumes BGR; does not handle a YUV frame
def _to_hsv(
        roi_bgr: np.ndarray
    ) -> np.ndarray:
    """BGR to OpenCV HSV (H in 0-179)."""
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

def _threshold_red(
        hsv: np.ndarray,
        ranges: HSVRanges
    ) -> np.ndarray:
    """Red mask: the union of red_low and red_high, since red straddles the hue wrap."""
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
    """0/255 mask for one band that doesn't wrap."""
    return cv2.inRange(hsv,
                       np.array(color_range.lower, dtype=np.uint8),
                       np.array(color_range.upper, dtype=np.uint8))

def _mean_hsv(
        hsv: np.ndarray,
        contour: np.ndarray
    ) -> tuple[float, float, float]:
    """Mean (H, S, V) inside a contour, to show where a blob sits against the bands. Trace only."""
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
    reject_counts: dict | None = None,
    trace: list[dict] | None = None,
    hsv: np.ndarray | None = None,
) -> list[TrafficLightCandidate]:
    """
    Find blobs in one color mask, run them through the area and aspect gates, and build candidates.

    Inputs:
        mask: 0/255 mask for one color.
        label: "red" | "yellow" | "green".
        reject_counts: Filled with seen / area / aspect / accepted. Every blob
            lands in exactly one bucket, so the buckets sum to "seen".
        trace: A list to receive one entry per blob that reached a gate
            (label, bbox, gate, area, aspect, fill, confidence, hsv), or None
            to skip. gate is None if accepted, else "area" or "aspect". Area
            rejects are traced only if their bbox clears TRACE_MIN_BBOX_FRAC.
        hsv: HSV image of the ROI, used only for the trace's mean HSV.

    Outputs:
        Accepted candidates.

    Side effects:
        Mutates reject_counts and trace.
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

        if area < blob_filter.min_area or area > blob_filter.max_area:
            rc["area"] += 1
            if trace is not None:
                bbox = cv2.boundingRect(contour)
                if bbox[2] * bbox[3] >= blob_filter.min_area * TRACE_MIN_BBOX_FRAC:
                    note(contour, "area", area, bbox)
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if h == 0:      # guard the w / h below
            rc["aspect"] += 1
            continue
        aspect = w / h

        # Scored before the aspect gate so aspect rejects still trace a confidence
        confidence = round(clamp(
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


def extract_traffic_light_candidates(
    roi: np.ndarray,
    hsv_ranges: HSVRanges,
    blob_filter: BlobFilter,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    trace: bool = False,
) -> tuple[list[TrafficLightCandidate], dict]:
    """
    Find traffic-light color candidates in the traffic ROI.

    Inputs:
        roi: (h, w, 3) uint8 BGR, e.g. ROICropResult.traffic_roi. Read-only
            views are fine.
        hsv_ranges: Normally from load_hsv_ranges(). Scaffold ranges are
            accepted for tuning; debug["calibrated"] reports which.
        trace: Also record every blob that reached a gate under
            debug["trace"]. Off by default; the live loop doesn't read it.

    Outputs:
        (candidates, debug). debug is for inspection; the pipeline doesn't
        pass it on. It always holds hsv, the red / yellow / green masks, roi
        (the input), mask_px ({color: nonzero px}), reject_counts ({color:
        {seen, area, aspect, accepted}}) and calibrated. With trace, it also
        holds trace (see _blobs_to_candidates).

    Raises:
        ValueError / TypeError: If roi is None, not uint8 or not (h, w, 3),
            or hsv_ranges is None.
    """
    if roi is None:
        raise ValueError("extract_traffic_light_candidates: received None")
    if roi.dtype != np.uint8:
        raise TypeError(f"extract_traffic_light_candidates: expected uint8, got {roi.dtype}")
    if roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError(f"extract_traffic_light_candidates: expected (H,W,3) BGR, got {roi.shape}")
    if hsv_ranges is None:
        raise ValueError(
            "extract_traffic_light_candidates: hsv_ranges is required. "
            "Load from calibration/hsv_ranges.json via load_hsv_ranges()."
        )

    hsv = _to_hsv(roi)

    masks = {
        RED: _threshold_red(hsv, hsv_ranges),
        YELLOW: _threshold_single(hsv, hsv_ranges.yellow),
        GREEN: _threshold_single(hsv, hsv_ranges.green),
    }

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
        RED: masks[RED],
        YELLOW: masks[YELLOW],
        GREEN: masks[GREEN],
        "roi": roi,
        "mask_px": {k: int(cv2.countNonZero(m)) for k, m in masks.items()},
        "reject_counts": reject_counts,
        "calibrated": hsv_ranges.is_calibrated,
    }
    if trace:
        debug["trace"] = trace_log

    return candidates, debug


def run_color_stage(
        roi: ROICropResult,
        config: ColorConfig = ColorConfig(),
        trace: bool = False,
    ) -> tuple[list[TrafficLightCandidate], dict]:
    """
    Stage entry point: run the color branch on one frame's traffic ROI.

    Inputs:
        roi: Supplies traffic_roi, the image analyzed, and the frame identity.
        config: Tuning. With hsv_ranges None the branch is off and the ROI
            is never read.
        trace: Record the per-blob trace (see extract_traffic_light_candidates).

    Outputs:
        Off: ([], {"enabled": False}). On: (candidates, debug) from
        extract_traffic_light_candidates, with debug["enabled"] True.
    """
    if config.hsv_ranges is None:
        return [], {"enabled": False}

    candidates, debug = extract_traffic_light_candidates(
        roi.traffic_roi,
        config.hsv_ranges,
        config.blob,
        roi.frame_id,
        roi.timestamp_ms,
        trace,
    )
    debug["enabled"] = True
    return candidates, debug


_LABEL_COLORS = {   # BGR
    RED: (0, 0, 255),
    YELLOW: (0, 200, 255),
    GREEN: (0, 200, 0),
}

def draw_candidates(
        roi_bgr: np.ndarray,
        candidates: list[TrafficLightCandidate]
    ) -> np.ndarray:
    """Copy of roi_bgr with each candidate's bbox and label drawn in its color (white if unknown)."""
    vis = roi_bgr.copy()
    for c in candidates:
        x, y, w, h = c.bbox
        color = _LABEL_COLORS.get(c.label, (255, 255, 255))
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.putText(vis, f"{c.label} {c.confidence:.2f}",
                    (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis