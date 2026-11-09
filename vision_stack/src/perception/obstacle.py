"""Obstacle detection: is something standing in the robot's path, and how far ahead is its base?

Purpose:
    The course is floor and white tape, neither with a color of its own, so
    a pixel much brighter than the floor, or clearly colored, is not floor.
    An opening wider than any tape removes the markings; what survives the
    gates is something solid (phase2_perception.md, "Stage 4C").

Main package:
    ObstacleResult: whether there is one, its base's distance in rows and
    cm, and whether it runs off the view bottom (clipped).
"""
from dataclasses import dataclass

import cv2
import numpy as np

from src.perception.ground import GroundHomography
from src.perception.roi_crop import ROICropResult
from src.perception.stop_line_table import StopLineTable

# In the order they're checked
GATE_SMALL, GATE_NARROW, GATE_SHORT, GATE_FAR = "small", "narrow", "short", "far"
GATES = (GATE_SMALL, GATE_NARROW, GATE_SHORT, GATE_FAR)


@dataclass(frozen=True)
class ObstacleConfig:
    """Pixel sizes are full-resolution px at 480x270, whatever step is."""
    # Of the lane ROI, which is wider than the path, so floor shows beside an
    # obstacle. Per frame: the floor runs from dark mats to light gray
    floor_percentile: float = 25.0
    min_contrast: float = 60.0      # gray levels above the floor that make a pixel not floor
    # |a - a_ref| + |b - b_ref| in Lab against the frame's neutral. 15: no false
    # alarm and every can frame on 2026-10-12; 20 lost 33 can frames
    min_chroma: float = 15.0
    min_value: int = 50             # brightest channel (HSV V) a colored pixel needs: color is noise in the dark
    # Wider than any lane tape (~42 px) and taller than any stop line (40)
    open_px: int = 47
    min_area_px: int = 2500         # blob area after the opening
    min_width_px: int = 60
    min_height_px: int = 60
    # ~30 cm; farther may be past the mat, and needs no stop
    max_rows_ahead: float = 90.0
    step: int = 2                   # every step-th pixel each way


@dataclass(frozen=True)
class ObstacleResult:
    """Rows grow downward from the lane ROI's bottom, as for the stop line."""
    detected: bool
    distance_px: float | None       # rows from the obstacle's base to the lane ROI bottom; 0 when clipped; None if not detected
    # None without a ground plane or table; clipped, the view bottom's cm ("this near or nearer")
    distance_cm: float | None
    clipped: bool                   # the blob runs off the ROI bottom: its base isn't in view
    bbox: tuple[int, int, int, int] | None      # (x, y, w, h) in obstacle-ROI px; add roi.obstacle_rect's origin for frame px
    area_px: int                    # the blob's pixels after the opening, in full-resolution px; 0 if not detected
    candidate_count: int            # blobs that survived the opening, before the gates
    frame_id: int
    timestamp_ms: int


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def _percentile(gray: np.ndarray, pct: float) -> float:
    """From a histogram, without a sort."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    return float(np.searchsorted(np.cumsum(hist), pct / 100.0 * gray.size))


def _floor_ab(roi: ROICropResult, k: int) -> tuple[float, float]:
    """
    The scene's neutral: the whole frame's median Lab (a, b), sampled sparsely.
    An object filling the lane ROI became its own reference there (2026-10-12).
    """
    ref, s = (roi.color_frame, 4 * k) if roi.color_frame is not None else (roi.obstacle_color_roi, k)
    lab = cv2.cvtColor(np.ascontiguousarray(ref[::s, ::s]), cv2.COLOR_BGR2Lab)
    return (_percentile(np.ascontiguousarray(lab[..., 1]), 50.0),
            _percentile(np.ascontiguousarray(lab[..., 2]), 50.0))


def _colored(roi: ROICropResult, color: np.ndarray, k: int, cfg: ObstacleConfig) -> np.ndarray:
    a_f, b_f = _floor_ab(roi, k)
    bgr = np.ascontiguousarray(color[::k, ::k])
    _, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab))
    chroma = cv2.add(cv2.absdiff(a, np.full_like(a, int(round(a_f)))),
                     cv2.absdiff(b, np.full_like(b, int(round(b_f)))))
    _, far = cv2.threshold(chroma, float(np.ceil(cfg.min_chroma)) - 1.0, 255, cv2.THRESH_BINARY)
    # HSV's V, not Lab's L: a dark blue can is bright in blue alone
    ch = cv2.split(bgr)
    _, lit = cv2.threshold(cv2.max(cv2.max(ch[0], ch[1]), ch[2]), float(cfg.min_value) - 1.0, 255,
                           cv2.THRESH_BINARY)
    return cv2.bitwise_and(far, lit)


def _masks(roi: ROICropResult, cfg: ObstacleConfig) -> tuple[float, np.ndarray, np.ndarray]:
    """(floor level, not-floor mask, opened mask) at 1 / step resolution."""
    gray, color = roi.obstacle_roi, roi.obstacle_color_roi
    if gray is None or color is None or roi.obstacle_rect is None:
        raise ValueError("obstacle: the ROI crop has no obstacle ROI (crop_rois always cuts one)")
    k = max(1, int(cfg.step))
    gray = np.ascontiguousarray(gray[::k, ::k])
    floor = _percentile(np.ascontiguousarray(roi.lane_roi[::k, ::k]), cfg.floor_percentile)
    # Whole-array OpenCV calls, not numpy boolean algebra: this runs every frame
    _, bright = cv2.threshold(gray, float(np.ceil(floor + cfg.min_contrast)) - 1.0, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(bright, _colored(roi, color, k, cfg))
    side = int(round(cfg.open_px / k))
    side += 1 - side % 2                    # odd, so the opening doesn't shift edges
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(side)) if side > 1 else mask
    return floor, mask, opened


def _blobs(opened: np.ndarray, step: int, roi_h: int) -> list[tuple[int, int, int, int, int]]:
    """(x, y, w, h, area) in full-resolution px; a blob on the last sampled row reads clipped."""
    k = max(1, int(step))
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    out = []
    for x, y, w, h, area in stats[1:, :5].tolist():
        bottom = roi_h if y + h >= opened.shape[0] else (y + h) * k
        out.append((x * k, y * k, w * k, bottom - y * k, area * k * k))
    return out


def _rows_ahead(roi: ROICropResult, base_y: int) -> float:
    lx, ly, lw, lh = roi.lane_rect
    return float(max(ly + lh - (roi.obstacle_rect[1] + base_y), 0))


def gate(w: int, h: int, area: int, rows: float, cfg: ObstacleConfig) -> str | None:
    """The first gate a blob fails, or None."""
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
    """From the homography at the base center, else the stop-line table; None without either."""
    size = tuple(roi.source_shape[:2])
    if ground is not None and size == (ground.image_size[1], ground.image_size[0]):
        ox, oy = roi.obstacle_rect[:2]
        _, y = ground.to_floor([(ox + base_x, oy + base_y)])[0]
        return round(float(y), 2) if y > 0.0 else 0.0     # not max(): it keeps -0.0
    if table is not None and size == (table.image_size[1], table.image_size[0]):
        return round(table.to_cm(rows), 2)
    return None


def _measure(roi, blob, count, ground, table) -> ObstacleResult:
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
    """Lower base first, then larger."""
    return (a[1] + a[3], a[4]) > (b[1] + b[3], b[4])


def _full_size(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    k_y, k_x = -(-h // mask.shape[0]), -(-w // mask.shape[1])
    return np.repeat(np.repeat(mask, k_y, axis=0), k_x, axis=1)[:h, :w]


def compute_obstacle(
        roi: ROICropResult,
        config: ObstacleConfig = ObstacleConfig(),
        ground: GroundHomography | None = None,
        table: StopLineTable | None = None,
    ) -> tuple[ObstacleResult, dict]:
    """(result, debug summary with the masks, blobs and gate counts). ValueError without an obstacle ROI."""
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
    """Production twin of compute_obstacle(), without the debug summary."""
    _, _, opened = _masks(roi, config)
    found = _blobs(opened, config.step, roi.obstacle_roi.shape[0])
    best = None
    for blob in found:
        x, y, w, h, area = blob
        if gate(w, h, area, _rows_ahead(roi, y + h), config) is None and (best is None or _nearer(blob, best)):
            best = blob
    return _nothing(len(found), roi) if best is None else _measure(roi, best, len(found), ground, table)
