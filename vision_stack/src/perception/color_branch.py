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
    3. Find blobs in each mask and gate them by area, aspect, roundness and
       a clipped core: a lit lamp is brighter than the camera can record, so
       its middle clips to near-white inside the colored ring. A shirt, a
       wall or a sign of the same color reflects light and never does.
    4. Score survivors by area and package them with the frame identity.

    Glow mode (ColorConfig.glow set): for lamps whose colored ring is no
    brighter than the same lens unlit, the course's bare LEDs (2026-10-06),
    where steps 3-4 read an unlit lens as lit. A lit LED clips the camera to
    white and an unlit lens never does, so the light starts there: the
    clipped-white spots, each named by the color band its ring of pixels
    falls in, and the spot with the most white pixels is the frame's one
    candidate (a traffic light shows one lamp at a time; the LED beats its
    dimmer reflection).
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
    """
    Blob gates for traffic-light candidates.

    Size, color and shape can't tell a lamp from a shirt of the same color;
    the core can. A lit lamp clips: inside its colored ring sit pixels the
    camera records as near-white (V at the top, S low), which is why every
    lamp measured on 2026-10-04 had a "missing" center in its color mask. A
    reflecting surface under room light stays colored throughout. A blob
    passes only with min_core_px such pixels inside its outline (the hole
    counts: the outline is the outer contour) and a round enough outline.
    """
    min_area: float = 30.0      # px^2; rejects mask speckle
    # px^2; a sanity bound, not what keeps junk out (the core and roundness
    # gates do). The lamps measured red 300-400, yellow 400-500, green 700-800
    # px^2 at normal exposure, the green glowing most (2026-10-04, with the
    # robot where it stops at the light); 1200 leaves headroom for stopping
    # closer. 600 rejected every green lamp; 300 the red ones too
    max_area: float = 1200.0
    min_aspect: float = 0.3     # w/h; together with max_aspect, rejects elongated streaks
    max_aspect: float = 3.0
    # px^2 scoring confidence 1.0: the smallest lamp at the stop (red, 300-400).
    # Confidence is (area - min_area) / (ref_area - min_area), so ref_area must
    # stay under max_area or no lamp can reach Phase 3's 0.40 gate (800 with a
    # 300 cap topped out at 0.35); at 350 every measured lamp scores 1.0, and
    # Phase 3's gate needs about 160 px^2
    ref_area: float = 350.0
    # Outline area over its enclosing circle's: ~0.9 for a disc or a ring
    # (the hole counts), 0.64 a square, 0.38 a 3:1 bar; rejects shirts, edges
    # and streaks that the aspect gate (bounding box only) lets through
    min_roundness: float = 0.5
    # The clipped core: pixels inside the outline with V >= core_min_v and
    # S <= core_max_s. The lamps' cores read V 254-255, S 4-5 (calibrate_lamps'
    # 5th percentiles, 2026-10-04). 0 turns the gate off
    core_min_v: int = 240
    core_max_s: int = 60
    min_core_px: int = 3

@dataclass(frozen=True)
class GlowFilter:
    """
    Glow mode's gates (see the module docstring). Measured on the course's
    light from its stop line, 2026-10-06: a lit LED's white center held 3-31
    clipped pixels (median 8-14), an unlit lens 0; with these, 196 of 196
    labelled frames read right, and an unlit lens can't be picked whatever
    its size.
    """
    # A clipped pixel: as BlobFilter's core gate. The lit LEDs' centers read
    # V 240-255, S 5-20; the board's glare clips too, so the ring decides
    white_min_v: int = 230
    white_max_s: int = 60
    min_white_px: int = 2       # fewer is a glint, not a lamp
    # A white spot's bounding box, width / height. Lit LED centers read 0.5-1.0;
    # the glare along the board's edge 10-11 (a 2 px tall strip) and its ring
    # picked up a red shirt below and won as red (2026-10-07). Outside these
    # it isn't a lamp
    min_white_aspect: float = 0.25
    max_white_aspect: float = 3.0
    # The ring a spot is named by: pixels within ring_px of it, in a color's
    # band. Its dimmer colored rim is 1-3 px wide at the stop
    ring_px: int = 2
    min_ring_px: int = 3        # ring pixels of the winning color, or the spot has no color
    # ... and this share of the ring. A lit LED's ring held 18-20% of its
    # color (2026-10-06 raw frames; the rest is washed-out glow under the
    # bands' S floors); a glare's ~60 px ring passed on a few stray colored
    # pixels, ~5%, where the mask panel showed nothing (2026-10-07)
    min_ring_share: float = 0.10
    # Confidence: white pixels over this, to 1.0. 2 px scores 0.40, Phase 3's gate
    ref_white_px: float = 5.0


@dataclass(frozen=True)
class ColorConfig:
    """Color tuning as one unit, so the stage takes a single config like every other stage."""
    hsv_ranges: HSVRanges | None = None     # None leaves the branch off
    blob: BlobFilter = field(default_factory=BlobFilter)
    glow: GlowFilter | None = None          # set: glow mode, the blob gates unused


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
        glow: GlowFilter | None = None,
    ) -> ColorConfig:
    """
    ColorConfig with calibrated ranges, ready to switch the branch on in a PipelineConfig.

    Inputs:
        json_path: Calibration file; see load_hsv_ranges().
        blob: Blob gates. None uses the BlobFilter() placeholders.
        glow: Glow mode's gates; None leaves the branch on the blob gates.
    """
    return ColorConfig(load_hsv_ranges(json_path), blob or BlobFilter(), glow)


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

def _roundness(
        contour: np.ndarray,
        area: float
    ) -> float:
    """The outline's area over its minimum enclosing circle's: 1.0 a perfect disc, lower the less round."""
    _, r = cv2.minEnclosingCircle(contour)
    return area / (np.pi * r * r) if r > 0 else 0.0

def _core_px(
        hsv: np.ndarray,
        contour: np.ndarray,
        blob_filter: BlobFilter
    ) -> int:
    """Clipped near-white pixels inside the outline, its hole included (see BlobFilter)."""
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
    The roundness and core gates, shared by both twins.

    Outputs:
        (gate, roundness, core_px): gate None if both pass, else "round" or
        "core"; core_px None when the core gate is off or wasn't reached.

    Raises:
        ValueError: If the core gate is on and hsv is None.
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
    Find blobs in one color mask, run them through the area, aspect, roundness and core gates, and build candidates.

    Inputs:
        mask: 0/255 mask for one color.
        label: "red" | "yellow" | "green".
        reject_counts: Filled with seen / area / aspect / round / core /
            accepted. Every blob lands in exactly one bucket, so the buckets
            sum to "seen".
        trace: A list to receive one entry per blob that reached a gate
            (label, bbox, gate, area, aspect, fill, confidence, roundness,
            core_px, hsv), or None to skip. gate is None if accepted, else
            "area", "aspect", "round" or "core". Area rejects are traced only
            if their bbox clears TRACE_MIN_BBOX_FRAC.
        hsv: HSV image of the ROI: the core gate reads it (required while
            blob_filter.min_core_px > 0), and the trace's mean HSV.

    Outputs:
        Accepted candidates.

    Side effects:
        Mutates reject_counts and trace.
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

        # Scored before the aspect gate so aspect rejects still trace a confidence
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
    """
    Production twin of _blobs_to_candidates(): same gates in the same order,
    without reject counting or tracing. hsv as there: the core gate's input.

    Outputs:
        Accepted candidates.
    """
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
    """1 where a pixel is clipped white: V at or over white_min_v, S at or under white_max_s."""
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
    Glow mode: the clipped-white spot with the most white pixels, named by its ring.

    Inputs:
        masks: The color bands' masks; a ring pixel counts for each band it's in.
        reject_counts: Filled per color with seen / white / shape / smaller /
            accepted (a spot without a ring color isn't a color's and isn't
            counted).
        trace: As _blobs_to_candidates, one entry per spot: area is its
            white pixels, gate None (the candidate), "white" (under
            min_white_px), "shape" (its box outside the white aspect
            limits), "ring" (no color, label "none") or "smaller" (another
            spot had more white). Each also holds ring (the ring's pixels)
            and votes ({color: ring pixels in its band}).

    Outputs:
        At most one candidate.
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
    Find traffic-light color candidates in the traffic ROI.

    Inputs:
        roi: (h, w, 3) uint8 BGR, e.g. ROICropResult.traffic_roi. Read-only
            views are fine.
        hsv_ranges: Normally from load_hsv_ranges(). Scaffold ranges are
            accepted for tuning; debug["calibrated"] reports which.
        trace: Also record every blob that reached a gate under
            debug["trace"]. Off by default; the live loop doesn't read it.
        glow: Glow mode's gates (_glow_candidates); None: the blob gates.

    Outputs:
        (candidates, debug). debug is for inspection; the pipeline doesn't
        pass it on. It always holds hsv, the red / yellow / green masks, roi
        (the input), mask_px ({color: nonzero px}), reject_counts ({color:
        {seen, area, aspect, round, core, accepted}}), calibrated and glow
        (glow mode on). In glow mode it also holds white (the clipped-white
        mask, 0/255; reject_counts per color are then seen / white / shape
        / smaller / accepted). With trace, it also holds trace (see
        _blobs_to_candidates, or _glow_candidates in glow mode).

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
    """
    Production twin of extract_traffic_light_candidates(): no debug dict
    (masks, pixel counts, reject counts) and no trace. glow as there.

    Outputs:
        Candidates, red then yellow then green.

    Raises:
        ValueError / TypeError: If roi is None, not uint8 or not (h, w, 3),
            or hsv_ranges is None.
    """
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
        config.glow,
    )
    debug["enabled"] = True
    return candidates, debug


def detect_color(
        roi: ROICropResult,
        config: ColorConfig = ColorConfig(),
    ) -> list[TrafficLightCandidate]:
    """
    Production twin of run_color_stage(): same stage, candidates only.

    Outputs:
        Candidates, or [] while the branch is off (hsv_ranges None).
    """
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
