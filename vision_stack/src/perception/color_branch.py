"""Color branch: traffic-light candidates from the traffic ROI.

Purpose:
    Color alone can't tell a lamp from a shirt of the same color, but a lit
    lamp clips the camera to white at its center, and the gates rest on that.
    MEASURED runs glow mode, tuned on the course's bare LEDs
    (phase2_perception.md, "Stage 3B").

Main package:
    TrafficLightCandidate: a color, a traffic-ROI bbox and a confidence.
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
    """One HSV band, bounds inclusive."""
    lower: tuple[int, int, int]   # (H_min, S_min, V_min)
    upper: tuple[int, int, int]   # (H_max, S_max, V_max)


@dataclass
class HSVRanges:
    """Hue in OpenCV units, 0-179. Red needs two bands because hue wraps. The defaults are a scaffold."""
    red_low: ColorRange = field(default_factory=lambda: ColorRange((0, 100, 100), (10, 255, 255)))
    red_high: ColorRange = field(default_factory=lambda: ColorRange((170, 100, 100), (180, 255, 255)))
    yellow: ColorRange = field(default_factory=lambda: ColorRange((20, 100, 100), (35, 255, 255)))
    green: ColorRange = field(default_factory=lambda: ColorRange((40, 100, 100), (80, 255, 255)))

    def __post_init__(self):
        self._validated = False   # True only after load_hsv_ranges() is used

    @property
    def is_calibrated(self) -> bool:
        """True only for ranges from load_hsv_ranges(); scaffold ranges still run, flagged in debug."""
        return self._validated


@dataclass
class BlobFilter:
    """
    A lit lamp has clipped near-white pixels inside its colored ring; a reflecting
    surface stays colored throughout. So a blob needs min_core_px of them inside its
    outline, and a round enough outline.
    """
    min_area: float = 30.0      # px^2; rejects mask speckle
    # A sanity bound: the core and roundness gates keep junk out. Lamps measured
    # 300-800 at the stop (2026-10-04)
    max_area: float = 1200.0
    min_aspect: float = 0.3     # w/h
    max_aspect: float = 3.0
    # Area scoring 1.0, the smallest lamp at the stop; must stay under max_area
    ref_area: float = 350.0
    # ~0.9 a disc or ring, 0.64 a square, 0.38 a 3:1 bar
    min_roundness: float = 0.5
    # Lamp cores read V 254-255, S 4-5 (2026-10-04); min_core_px 0 turns the gate off
    core_min_v: int = 240
    core_max_s: int = 60
    min_core_px: int = 3


@dataclass(frozen=True)
class GlowFilter:
    """Measured on the course's light from its stop line (2026-10-06): 196 of 196 frames read right."""
    # Lit LED centers read V 240-255, S 5-20; glare clips too, so the ring decides
    white_min_v: int = 230
    white_max_s: int = 60
    min_white_px: int = 2       # fewer is a glint
    # Box w / h: LED centers read 0.5-1.0; the board-edge glare 10-11 won as red (2026-10-07)
    min_white_aspect: float = 0.25
    max_white_aspect: float = 3.0
    # The colored rim is 1-3 px wide at the stop
    ring_px: int = 2
    min_ring_px: int = 3        # of the winning color, else the spot has no color
    # A lit LED's ring held 18-20% of its color; glare's ~5% (2026-10-06/07)
    min_ring_share: float = 0.10
    # Confidence is white px / this, to 1.0: 2 px scores 0.40, Phase 3's gate
    ref_white_px: float = 5.0


@dataclass(frozen=True)
class ColorConfig:
    hsv_ranges: HSVRanges | None = None     # None leaves the branch off
    blob: BlobFilter = field(default_factory=BlobFilter)
    glow: GlowFilter | None = None          # set: glow mode, the blob gates unused


@dataclass
class TrafficLightCandidate:
    label: str                          # "red" | "yellow" | "green"
    bbox: tuple[int, int, int, int]     # (x, y, w, h) in traffic-ROI px
    # Area only, so fusion's pick is the largest blob of any color
    confidence: float
    frame_id: int
    timestamp_ms: int


_HSV_CAPS = (180, 255, 255)     # H allows 180 as an upper bound for the red wrap


def _band(data: dict, key: str) -> ColorRange:
    """Checked, so a typo fails at load, not on frame 1."""
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
    Calibrated ranges from {"red_low", "red_high", "yellow", "green": {"lower", "upper": [H, S, V]}}.
    KeyError or ValueError names a missing or out-of-range band.
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
        glow: GlowFilter | None = None,
    ) -> ColorConfig:
    """glow=None leaves the branch on the blob gates."""
    return ColorConfig(load_hsv_ranges(json_path), blob or BlobFilter(), glow)


def _to_hsv(
        roi_bgr: np.ndarray
    ) -> np.ndarray:
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)


def _threshold_red(
        hsv: np.ndarray,
        ranges: HSVRanges
    ) -> np.ndarray:
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
    return cv2.inRange(hsv,
                       np.array(color_range.lower, dtype=np.uint8),
                       np.array(color_range.upper, dtype=np.uint8))


def _mean_hsv(
        hsv: np.ndarray,
        contour: np.ndarray
    ) -> tuple[float, float, float]:
    """Trace only: where a blob sits against the bands."""
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([[[x, y]]])], -1, 255,
                     thickness=cv2.FILLED)
    m = cv2.mean(hsv[y:y + h, x:x + w], mask=mask)
    return (round(m[0], 1), round(m[1], 1), round(m[2], 1))


def _roundness(
        contour: np.ndarray,
        area: float
    ) -> float:
    """Outline area over its enclosing circle's; 1.0 is a disc."""
    _, r = cv2.minEnclosingCircle(contour)
    return area / (np.pi * r * r) if r > 0 else 0.0


def _core_px(
        hsv: np.ndarray,
        contour: np.ndarray,
        blob_filter: BlobFilter
    ) -> int:
    """Clipped pixels inside the outline, its hole included."""
    x, y, w, h = cv2.boundingRect(contour)
    inside = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(inside, [contour - np.array([[[x, y]]])], -1, 255, thickness=cv2.FILLED)
    patch = hsv[y:y + h, x:x + w]
    clipped = (patch[..., 2] >= blob_filter.core_min_v) & (patch[..., 1] <= blob_filter.core_max_s)
    return int(np.count_nonzero(clipped & (inside > 0)))


def _shape_and_core(
        contour: np.ndarray,
        area: float,
        hsv: np.ndarray | None,
        blob_filter: BlobFilter
    ) -> tuple[str | None, float, int | None]:
    """
    (gate or None, roundness, core_px), shared by both twins. ValueError if the core gate is on without hsv.
    """
    roundness = _roundness(contour, area)
    if roundness < blob_filter.min_roundness:
        return "round", roundness, None
    if blob_filter.min_core_px <= 0:
        return None, roundness, None
    if hsv is None:
        raise ValueError("the core gate needs the ROI's HSV image (BlobFilter.min_core_px > 0)")
    core = _core_px(hsv, contour, blob_filter)
    return (None if core >= blob_filter.min_core_px else "core"), roundness, core

# Area rejects are traced only if their bbox reaches this share of min_area
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
    Accepted candidates from one color's mask. Every blob lands in one reject_counts
    bucket; trace gets each gated blob. hsv feeds the core gate and the trace.
    """
    candidates = []

    rc = reject_counts if reject_counts is not None else {}
    for _k in ("seen", "area", "aspect", "round", "core", "accepted"):
        rc.setdefault(_k, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def note(contour, gate, area, bbox, aspect=None, confidence=None, roundness=None, core=None):
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
            "roundness": None if roundness is None else round(roundness, 3),
            "core_px": core,
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

        confidence = round(clamp(
            (area - blob_filter.min_area) / max(blob_filter.ref_area - blob_filter.min_area, 1.0),
            0.0, 1.0
        ), 4)

        if aspect < blob_filter.min_aspect or aspect > blob_filter.max_aspect:
            rc["aspect"] += 1
            note(contour, "aspect", area, (x, y, w, h), aspect, confidence)
            continue

        gate, roundness, core = _shape_and_core(contour, area, hsv, blob_filter)
        if gate is not None:
            rc[gate] += 1
            note(contour, gate, area, (x, y, w, h), aspect, confidence, roundness, core)
            continue

        rc["accepted"] += 1
        note(contour, None, area, (x, y, w, h), aspect, confidence, roundness, core)
        candidates.append(TrafficLightCandidate(
            label = label,
            bbox = (x, y, w, h),
            confidence = confidence,
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
        ))

    return candidates


def _filter_blobs(
    mask: np.ndarray,
    label: str,
    blob_filter: BlobFilter,
    frame_id: int,
    timestamp_ms: int,
    hsv: np.ndarray | None = None,
) -> list[TrafficLightCandidate]:
    """Production twin of _blobs_to_candidates(), without counts or trace."""
    candidates = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < blob_filter.min_area or area > blob_filter.max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if h == 0:      # guard the w / h below
            continue
        aspect = w / h
        if aspect < blob_filter.min_aspect or aspect > blob_filter.max_aspect:
            continue
        if _shape_and_core(contour, area, hsv, blob_filter)[0] is not None:
            continue

        confidence = round(clamp(
            (area - blob_filter.min_area) / max(blob_filter.ref_area - blob_filter.min_area, 1.0),
            0.0, 1.0
        ), 4)

        candidates.append(TrafficLightCandidate(
            label = label,
            bbox = (x, y, w, h),
            confidence = confidence,
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
        ))

    return candidates


def _masks(hsv: np.ndarray, ranges: HSVRanges) -> dict:
    return {RED: _threshold_red(hsv, ranges), YELLOW: _threshold_single(hsv, ranges.yellow),
            GREEN: _threshold_single(hsv, ranges.green)}


def _white_mask(hsv: np.ndarray, glow: GlowFilter) -> np.ndarray:
    return ((hsv[..., 2] >= glow.white_min_v) & (hsv[..., 1] <= glow.white_max_s)).astype(np.uint8)


def _glow_candidates(
    hsv: np.ndarray,
    masks: dict,
    glow: GlowFilter,
    frame_id: int,
    timestamp_ms: int,
    reject_counts: dict | None = None,
    trace: list[dict] | None = None,
) -> list[TrafficLightCandidate]:
    """
    At most one candidate: the clipped-white spot with the most white, named by the
    band holding the most of its ring. A spot without a ring color isn't counted for any color.
    """
    white = _white_mask(hsv, glow)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * glow.ring_px + 1,) * 2)
    rc = reject_counts if reject_counts is not None else {}
    for c in masks:
        for k in ("seen", "white", "shape", "smaller", "accepted"):
            rc.setdefault(c, {}).setdefault(k, 0)
    spots = []
    for k in range(1, n):
        x, y, w, h, px = (int(v) for v in stats[k])
        y0, y1 = max(0, y - glow.ring_px), y + h + glow.ring_px
        x0, x1 = max(0, x - glow.ring_px), x + w + glow.ring_px
        spot = (labels[y0:y1, x0:x1] == k).astype(np.uint8)
        ring = cv2.dilate(spot, kernel) > spot
        votes = {c: int(np.count_nonzero(ring & (m[y0:y1, x0:x1] > 0))) for c, m in masks.items()}
        label = max(votes, key=votes.get)
        if votes[label] < max(glow.min_ring_px, glow.min_ring_share * np.count_nonzero(ring)):
            label = None
        aspect = w / h
        if px < glow.min_white_px:
            gate = "white"
        elif not glow.min_white_aspect <= aspect <= glow.max_white_aspect:
            gate = "shape"
        else:
            gate = "ring" if label is None else None
        bbox = (x0, y0, min(x1, hsv.shape[1]) - x0, min(y1, hsv.shape[0]) - y0)
        spots.append([label, bbox, gate, px, int(np.count_nonzero(ring)), votes])
    eligible = [sp for sp in spots if sp[2] is None]
    best = max(eligible, key=lambda sp: sp[3]) if eligible else None
    for sp in eligible:
        if sp is not best:
            sp[2] = "smaller"
    for label, bbox, gate, px, ring_n, votes in spots:
        if label is not None:
            rc[label]["seen"] += 1
            rc[label]["accepted" if gate is None else gate] += 1
        if trace is not None:
            x, y, w, h = bbox
            trace.append({"label": label or "none", "bbox": bbox, "gate": gate, "area": float(px),
                          "aspect": round(w / h, 3) if h else None, "fill": round(px / (w * h), 3) if w * h else None,
                          "confidence": round(min(1.0, px / glow.ref_white_px), 4), "roundness": None,
                          "core_px": px, "hsv": None, "ring": ring_n, "votes": votes})
    if best is None:
        return []
    return [TrafficLightCandidate(label=best[0], bbox=best[1], confidence=round(min(1.0, best[3] / glow.ref_white_px), 4),
                                  frame_id=frame_id, timestamp_ms=timestamp_ms)]


def extract_traffic_light_candidates(
    roi: np.ndarray,
    hsv_ranges: HSVRanges,
    blob_filter: BlobFilter,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    trace: bool = False,
    glow: GlowFilter | None = None,
) -> tuple[list[TrafficLightCandidate], dict]:
    """
    (candidates, debug with the masks and counts; trace if asked). ValueError or
    TypeError if roi or hsv_ranges is malformed.
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
    masks = _masks(hsv, hsv_ranges)

    reject_counts = {}
    trace_log = [] if trace else None
    if glow is not None:
        candidates = _glow_candidates(hsv, masks, glow, frame_id, timestamp_ms, reject_counts, trace_log)
    else:
        candidates = []
        for label, mask in masks.items():
            rc = reject_counts[label] = {}
            candidates += _blobs_to_candidates(
                mask, label, blob_filter, frame_id, timestamp_ms,
                rc, trace_log, hsv,
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
        "glow": glow is not None,
    }
    if glow is not None:
        debug["white"] = _white_mask(hsv, glow) * 255
    if trace:
        debug["trace"] = trace_log

    return candidates, debug


def find_traffic_light_candidates(
    roi: np.ndarray,
    hsv_ranges: HSVRanges,
    blob_filter: BlobFilter,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    glow: GlowFilter | None = None,
) -> list[TrafficLightCandidate]:
    """Production twin of extract_traffic_light_candidates(): red, then yellow, then green."""
    if roi is None:
        raise ValueError("find_traffic_light_candidates: received None")
    if roi.dtype != np.uint8:
        raise TypeError(f"find_traffic_light_candidates: expected uint8, got {roi.dtype}")
    if roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError(f"find_traffic_light_candidates: expected (H,W,3) BGR, got {roi.shape}")
    if hsv_ranges is None:
        raise ValueError(
            "find_traffic_light_candidates: hsv_ranges is required. "
            "Load from calibration/hsv_ranges.json via load_hsv_ranges()."
        )

    hsv = _to_hsv(roi)
    masks = _masks(hsv, hsv_ranges)
    if glow is not None:
        return _glow_candidates(hsv, masks, glow, frame_id, timestamp_ms)

    candidates = []
    for label, mask in masks.items():
        candidates += _filter_blobs(mask, label, blob_filter, frame_id, timestamp_ms, hsv)

    return candidates


def run_color_stage(
        roi: ROICropResult,
        config: ColorConfig = ColorConfig(),
        trace: bool = False,
    ) -> tuple[list[TrafficLightCandidate], dict]:
    """Off (hsv_ranges None): ([], {"enabled": False}), without reading the ROI."""
    if config.hsv_ranges is None:
        return [], {"enabled": False}

    candidates, debug = extract_traffic_light_candidates(
        roi.traffic_roi,
        config.hsv_ranges,
        config.blob,
        roi.frame_id,
        roi.timestamp_ms,
        trace,
        config.glow,
    )
    debug["enabled"] = True
    return candidates, debug


def detect_color(
        roi: ROICropResult,
        config: ColorConfig = ColorConfig(),
    ) -> list[TrafficLightCandidate]:
    """Production twin of run_color_stage(); [] while the branch is off."""
    if config.hsv_ranges is None:
        return []

    return find_traffic_light_candidates(
        roi.traffic_roi,
        config.hsv_ranges,
        config.blob,
        roi.frame_id,
        roi.timestamp_ms,
        config.glow,
    )

_LABEL_COLORS = {   # BGR
    RED: (0, 0, 255),
    YELLOW: (0, 200, 255),
    GREEN: (0, 200, 0),
}


def draw_candidates(
        roi_bgr: np.ndarray,
        candidates: list[TrafficLightCandidate]
    ) -> np.ndarray:
    vis = roi_bgr.copy()
    for c in candidates:
        x, y, w, h = c.bbox
        color = _LABEL_COLORS.get(c.label, (255, 255, 255))
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.putText(vis, f"{c.label} {c.confidence:.2f}",
                    (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis
