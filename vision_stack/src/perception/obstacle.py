"""Obstacle detection: is something standing in the robot's path, and how far ahead is its base?

Purpose:
    The course is dark mats and white tape, and neither has any color. So in
    the obstacle ROI (roi_crop.OBSTACLE, the middle of the frame from the
    bottom of the view up past the lane ROI), a pixel that is much brighter
    than the floor, or clearly colored, is "not floor". Tape is bright too,
    but thin: a morphological opening with a square wider than any tape
    (lane lines, dashes, stop lines) removes it, and what survives is
    something solid and large. Each surviving blob is gated on size and on
    how far ahead its base is, and the nearest one that passes is the
    obstacle.

    Upright objects rise above the floor, so the blob's lowest row is where
    the object meets the floor. The floor at that row is measured the way
    the stop line is (stop_line_distance.py): rows above the lane ROI
    bottom (distance_px), and cm through the ground homography or, without
    one, the stop-line table, whose curve maps any floor row to cm. The
    table's reference is the robot's front edge, and its row 0 (the bottom
    of the view) is ~10.2 cm ahead of it (2026-09-30), so nothing nearer
    than that has a visible base: a blob that runs off the ROI bottom is
    clipped, "at the edge of the view or nearer".

    The loop already runs at its 50 ms budget on the Pi, so the masks are
    built on every step-th pixel each way (2: a quarter of the pixels). An
    obstacle is tens of pixels across, so nothing it needs is lost; every
    size and row in the config and the result stays in full-resolution px.

    What it can't see: an object no brighter than the mats and with no
    color (black or dark gray), and an object wholly outside the middle of
    the frame. What it may mistake for one: a bright or colored patch on
    the floor at least open_px across (glare, a light-colored floor past the
    mat's edge, a stop sign low in the corridor). The gates are a first
    guess; tune them on the course with the obstacle view
    (make phase2 VIEWS=obstacle).

Main package:
    ObstacleConfig: the not-floor thresholds, the opening and the blob gates.
    ObstacleResult: whether there is an obstacle, its distance in rows and
        cm, whether it runs off the view bottom, its box, and the frame stamp.
        One per frame; detected False when there is none.
    compute_obstacle(): the debug version, with the masks and every blob's fate.
    estimate_obstacle(): the production twin, the same result without them.
    gate(): the blob gates, shared by both.

Flow:
    1. Take every step-th pixel of the ROI, gray and color.
    2. Floor level: a low percentile of the gray (from its histogram),
       capped at floor_cap so an object filling the view can't become "the floor".
    3. Not floor: gray >= floor + min_contrast, or HSV S >= min_saturation
       with V >= min_value.
    4. Opening with an open_px square: tape and specks go.
    5. Connected blobs, scaled back to full-resolution px; each gated: area,
       width, height, then how far ahead its base is.
    6. The nearest passing blob (lowest base) is measured: distance_px from
       its base to the lane ROI bottom; distance_cm at its base's center.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from src.perception.ground import GroundHomography
from src.perception.roi_crop import ROICropResult
from src.perception.stop_line_table import StopLineTable

# Blob gates, in the order they're checked; the debug summary counts each
GATE_SMALL, GATE_NARROW, GATE_SHORT, GATE_FAR = "small", "narrow", "short", "far"
GATES = (GATE_SMALL, GATE_NARROW, GATE_SHORT, GATE_FAR)


# =============================================================================
# Config and result
# =============================================================================

@dataclass(frozen=True)
class ObstacleConfig:
    """
    Tuning for the obstacle detector. Pixel sizes are full-resolution px at
    480x270, whatever step is; first guesses from the course's dimensions,
    not yet tuned on course frames.
    """
    floor_percentile: float = 25.0  # the floor's gray level: this percentile of the ROI
    # Cap on that level: mats are dark, so a "floor" brighter than this is
    # an object filling the ROI, and measuring contrast from it would lose it
    floor_cap: float = 70.0
    min_contrast: float = 60.0      # gray levels above the floor that make a pixel not floor
    min_saturation: int = 80        # HSV S (0-255) that makes a pixel colored; mats and tape are gray
    min_value: int = 50             # HSV V a colored pixel needs: hue and saturation are noise in the dark
    # Side of the opening square. Wider than any tape in the lane ROI (lane
    # lines measure up to ~42 px, LaneOffsetConfig.max_width_px 45 in
    # MEASURED) and taller than any stop line (StopLineFilter max_thickness_px
    # 40), so the course's markings never survive it
    open_px: int = 47
    min_area_px: int = 2500         # blob area after the opening
    min_width_px: int = 60
    min_height_px: int = 60
    # Farthest base counted, in rows above the lane ROI bottom: ~30 cm on the
    # 2026-09-30 stop-line table. Past it the floor may be past the mat's
    # edge, and nothing that far needs a stop
    max_rows_ahead: float = 90.0
    step: int = 2                   # work on every step-th pixel each way; 1 is full resolution


@dataclass(frozen=True)
class ObstacleResult:
    """
    The nearest obstacle in one frame. frame_id and timestamp_ms are carried
    from capture, never re-derived.

    Rows grow downward, toward the robot; the reference row is the bottom of
    the lane ROI (the bottom of the view), as for the stop line.
    """
    detected: bool
    distance_px: float | None       # rows from the obstacle's base to the lane ROI bottom; 0 when clipped; None if not detected
    # Floor cm ahead of the ground homography's reference, or of the stop-line
    # table's (the robot's front edge); None without either. Clipped: the
    # cm at the view bottom, i.e. "this near or nearer"
    distance_cm: float | None
    clipped: bool                   # the blob runs off the ROI bottom: its base isn't in view
    bbox: tuple[int, int, int, int] | None      # (x, y, w, h) in obstacle-ROI px; add roi.obstacle_rect's origin for frame px
    area_px: int                    # the blob's pixels after the opening, in full-resolution px; 0 if not detected
    candidate_count: int            # blobs that survived the opening, before the gates
    frame_id: int
    timestamp_ms: int


# =============================================================================
# Stages shared by both versions
# =============================================================================

def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

def _percentile(gray: np.ndarray, pct: float) -> float:
    """The lowest gray level with at least pct% of the pixels at or below it, from a histogram: no sort."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    return float(np.searchsorted(np.cumsum(hist), pct / 100.0 * gray.size))

def _masks(roi: ROICropResult, cfg: ObstacleConfig) -> tuple[float, np.ndarray, np.ndarray]:
    """
    (floor level, not-floor mask, mask after the opening), at 1 / step
    resolution; masks are uint8 0 / 255.
    """
    gray, color = roi.obstacle_roi, roi.obstacle_color_roi
    if gray is None or color is None or roi.obstacle_rect is None:
        raise ValueError("obstacle: the ROI crop has no obstacle ROI (crop_rois always cuts one)")
    k = max(1, int(cfg.step))
    gray = np.ascontiguousarray(gray[::k, ::k])
    floor = min(_percentile(gray, cfg.floor_percentile), cfg.floor_cap)
    hsv = cv2.cvtColor(np.ascontiguousarray(color[::k, ::k]), cv2.COLOR_BGR2HSV)
    # Whole-array OpenCV calls rather than numpy boolean algebra: this runs
    # every frame on the Pi. A pixel is bright at gray >= floor + min_contrast
    _, bright = cv2.threshold(gray, float(np.ceil(floor + cfg.min_contrast)) - 1.0, 255, cv2.THRESH_BINARY)
    colored = cv2.inRange(hsv, (0, cfg.min_saturation, cfg.min_value), (255, 255, 255))
    mask = cv2.bitwise_or(bright, colored)
    side = int(round(cfg.open_px / k))
    side += 1 - side % 2                    # odd, so the square is centered on its pixel and the opening doesn't shift edges
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(side)) if side > 1 else mask
    return floor, mask, opened

def _blobs(opened: np.ndarray, step: int, roi_h: int) -> list[tuple[int, int, int, int, int]]:
    """
    Each blob's (x, y, w, h, area) in full-resolution obstacle-ROI px,
    background dropped. A blob on the last sampled row runs to the ROI's
    own bottom, so it reads clipped whatever the step.
    """
    k = max(1, int(step))
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    out = []
    for x, y, w, h, area in stats[1:, :5].tolist():
        bottom = roi_h if y + h >= opened.shape[0] else (y + h) * k
        out.append((x * k, y * k, w * k, bottom - y * k, area * k * k))
    return out

def _rows_ahead(roi: ROICropResult, base_y: int) -> float:
    """Rows from an obstacle-ROI row to the lane ROI bottom (the view's), never negative."""
    lx, ly, lw, lh = roi.lane_rect
    return float(max(ly + lh - (roi.obstacle_rect[1] + base_y), 0))

def gate(w: int, h: int, area: int, rows: float, cfg: ObstacleConfig) -> str | None:
    """The first gate a blob fails, in GATES order; None if it passes them all."""
    if area < cfg.min_area_px:
        return GATE_SMALL
    if w < cfg.min_width_px:
        return GATE_NARROW
    if h < cfg.min_height_px:
        return GATE_SHORT
    if rows > cfg.max_rows_ahead:
        return GATE_FAR
    return None

def _cm(
        roi: ROICropResult,
        rows: float,
        base_x: float,
        base_y: int,
        ground: GroundHomography | None,
        table: StopLineTable | None,
    ) -> float | None:
    """
    Floor cm at the base: forward of the homography's reference at the base
    center, else the stop-line table at its rows; None without either, or
    when the frame isn't the size it was fit at.
    """
    size = tuple(roi.source_shape[:2])
    if ground is not None and size == (ground.image_size[1], ground.image_size[0]):
        ox, oy = roi.obstacle_rect[:2]
        _, y = ground.to_floor([(ox + base_x, oy + base_y)])[0]
        return round(float(y), 2) if y > 0.0 else 0.0     # not max(): it keeps -0.0
    if table is not None and size == (table.image_size[1], table.image_size[0]):
        return round(table.to_cm(rows), 2)
    return None

def _measure(roi, blob, count, ground, table) -> ObstacleResult:
    """ObstacleResult for one blob (x, y, w, h, area)."""
    x, y, w, h, area = blob
    base_y = y + h                          # the row just below the blob: its near edge
    clipped = base_y >= roi.obstacle_roi.shape[0]
    rows = _rows_ahead(roi, base_y)
    return ObstacleResult(
        detected = True,
        distance_px = 0.0 if clipped else rows,
        distance_cm = _cm(roi, 0.0 if clipped else rows, x + w / 2.0, base_y, ground, table),
        clipped = clipped,
        bbox = (x, y, w, h),
        area_px = area,
        candidate_count = count,
        frame_id = roi.frame_id,
        timestamp_ms = roi.timestamp_ms,
    )

def _nothing(count: int, roi: ROICropResult) -> ObstacleResult:
    return ObstacleResult(
        detected = False, distance_px = None, distance_cm = None, clipped = False, bbox = None,
        area_px = 0, candidate_count = count, frame_id = roi.frame_id, timestamp_ms = roi.timestamp_ms,
    )

def _nearer(a, b) -> bool:
    """Blob a is nearer than b: lower base, then larger."""
    return (a[1] + a[3], a[4]) > (b[1] + b[3], b[4])

def _full_size(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A 1 / step mask back at the ROI's size, each sample repeated over the pixels it stood for."""
    h, w = shape[:2]
    k_y, k_x = -(-h // mask.shape[0]), -(-w // mask.shape[1])
    return np.repeat(np.repeat(mask, k_y, axis=0), k_x, axis=1)[:h, :w]


# =============================================================================
# Entry points
# =============================================================================

def compute_obstacle(
        roi: ROICropResult,
        config: ObstacleConfig = ObstacleConfig(),
        ground: GroundHomography | None = None,
        table: StopLineTable | None = None,
    ) -> tuple[ObstacleResult, dict]:
    """
    Find the nearest obstacle in the robot's path.

    Inputs:
        roi: From crop_rois(): the obstacle ROI (gray and color) and its
            rect, the lane ROI's rect (the reference row), the frame size
            and the frame stamp.
        config: The thresholds and gates.
        ground: The ground homography (PipelineConfig.ground).
        table: The stop-line table (PipelineConfig.stop_line_table), used
            when there's no ground homography. Neither leaves distance_cm None.

    Outputs:
        (result, debug_summary). debug_summary holds frame_id, timestamp_ms,
        floor_level, threshold (the gray level a bright pixel needs),
        not_floor_px (full-resolution px), mask and opened (uint8 images at
        the obstacle ROI's size), blobs (one dict each: bbox, area,
        rows_ahead, gate; None = passed), reject_counts (per gate, plus
        "passed") and log.

    Raises:
        ValueError: If roi carries no obstacle ROI.
    """
    floor, mask, opened = _masks(roi, config)
    k, roi_h = max(1, int(config.step)), roi.obstacle_roi.shape[0]
    found = _blobs(opened, k, roi_h)
    blobs, counts, log, best = [], {g: 0 for g in GATES} | {"passed": 0}, [], None
    for blob in found:
        x, y, w, h, area = blob
        rows = _rows_ahead(roi, y + h)
        why = gate(w, h, area, rows, config)
        blobs.append({"bbox": (x, y, w, h), "area": area, "rows_ahead": rows, "gate": why})
        counts[why or "passed"] += 1
        if why is not None:
            log.append(f"[REJECT] blob at x={x} y={y} {w}x{h} area {area}: {why}")
        elif best is None or _nearer(blob, best):
            best = blob
    if best is None:
        result = _nothing(len(found), roi)
    else:
        result = _measure(roi, best, len(found), ground, table)
        cm = "" if result.distance_cm is None else f" = {result.distance_cm:.1f}cm"
        where = "at the view bottom" if result.clipped else f"{result.distance_px:.0f}px{cm} ahead"
        log.append(f"[OBSTACLE] {result.bbox[2]}x{result.bbox[3]} {where}")
    return result, {
        "frame_id": roi.frame_id,
        "timestamp_ms": roi.timestamp_ms,
        "floor_level": floor,
        "threshold": floor + config.min_contrast,
        "not_floor_px": int(np.count_nonzero(mask)) * k * k,
        "mask": _full_size(mask, roi.obstacle_roi.shape),
        "opened": _full_size(opened, roi.obstacle_roi.shape),
        "blobs": blobs,
        "reject_counts": counts,
        "log": log,
    }


def estimate_obstacle(
        roi: ROICropResult,
        config: ObstacleConfig = ObstacleConfig(),
        ground: GroundHomography | None = None,
        table: StopLineTable | None = None,
    ) -> ObstacleResult:
    """
    Production twin of compute_obstacle(): same result, no debug summary, log or blob records.

    Raises:
        ValueError: If roi carries no obstacle ROI.
    """
    _, _, opened = _masks(roi, config)
    found = _blobs(opened, config.step, roi.obstacle_roi.shape[0])
    best = None
    for blob in found:
        x, y, w, h, area = blob
        if gate(w, h, area, _rows_ahead(roi, y + h), config) is None and (best is None or _nearer(blob, best)):
            best = blob
    return _nothing(len(found), roi) if best is None else _measure(roi, best, len(found), ground, table)
