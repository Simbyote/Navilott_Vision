"""Stop-line distance: how far ahead of the robot the nearest stop line is.

Purpose:
    Geometry only detects; this turns the stop-line candidates into one
    distance per frame. distance_px needs no calibration; distance_cm comes
    from the ground homography, else the stop-line table.

Main package:
    StopLineResult: one per frame, detected False when there's no line.
"""
import math
from dataclasses import dataclass

from src.perception.geometry import GeometryBranchResult, StopLineCandidate
from src.perception.ground import GroundHomography
from src.perception.roi_crop import ROICropResult
from src.perception.stop_line_table import StopLineTable
from src.utils import check_same_frame


@dataclass(frozen=True)
class StopLineDistanceConfig:
    min_confidence: float = 0.4     # candidates below this aren't measured


@dataclass(frozen=True)
class StopLineResult:
    """Rows grow downward, toward the robot; the reference is the lane ROI's bottom, the view's edge."""
    detected: bool
    distance_px: float | None       # rows from the line's nearest point to the lane ROI bottom; 0 when on it; None if not detected
    y_near_px: float | None         # lane-ROI row of the line's nearest point
    x_left: float | None            # lane-ROI px; add roi.lane_rect[0] for frame x
    x_right: float | None
    tilt_deg: float | None          # + = right end nearer the robot
    clipped: bool                   # the line runs off the ROI bottom: the robot is on it
    confidence: float               # the candidate's; 0.0 if not detected
    candidate_count: int            # candidates geometry found, before the confidence gate
    frame_id: int
    timestamp_ms: int
    # cm ahead along the robot's centerline; 0 on the line, None without a ground plane or table
    distance_cm: float | None = None
    proximity: float = 0.0          # [0, 1] y_near_px / ROI height; 1.0 = at the ROI bottom


def _nothing(count: int, frame_id: int, timestamp_ms: int) -> StopLineResult:
    return StopLineResult(
        detected = False, distance_px = None, y_near_px = None, x_left = None,
        x_right = None, tilt_deg = None, clipped = False, confidence = 0.0,
        candidate_count = count, frame_id = frame_id, timestamp_ms = timestamp_ms,
        distance_cm = None, proximity = 0.0,
    )


def _table_cm(c: StopLineCandidate, roi: ROICropResult, table: StopLineTable | None) -> float | None:
    """0 for a clipped line; None without a table or at another frame size."""
    if table is None or tuple(roi.source_shape[:2]) != (table.image_size[1], table.image_size[0]):
        return None
    if c.clipped:
        return 0.0
    return round(table.to_cm(max(roi.lane_rect[3] - c.y_near_px, 0.0)), 2)


def _distance_cm(
        c: StopLineCandidate,
        roi: ROICropResult,
        ground: GroundHomography | None,
        table: StopLineTable | None = None,
    ) -> float | None:
    """
    Floor cm to where the near edge crosses the robot's centerline, rebuilt from
    its midpoint row and tilt. Without a usable ground plane, the table's cm.
    """
    if ground is None or tuple(roi.source_shape[:2]) != (ground.image_size[1], ground.image_size[0]):
        return _table_cm(c, roi, table)
    rx, ry, _, roi_h = roi.lane_rect
    mid = (c.x_left + c.x_right) / 2.0
    slope = math.tan(math.radians(c.tilt_deg))
    edge = (lambda x: float(roi_h)) if c.clipped else (lambda x: c.y_bottom_px + (x - mid) * slope)
    y = ground.forward_at_centerline((rx + c.x_left, ry + edge(c.x_left)),
                                     (rx + c.x_right, ry + edge(c.x_right)))
    return None if y is None else round(max(y, 0.0), 2)


def _measure(
        c: StopLineCandidate,
        roi: ROICropResult,
        ground: GroundHomography | None,
        count: int,
        table: StopLineTable | None = None,
    ) -> StopLineResult:
    roi_h = roi.lane_rect[3]
    return StopLineResult(
        detected = True,
        distance_px = round(max(roi_h - c.y_near_px, 0.0), 2),
        y_near_px = c.y_near_px,
        x_left = c.x_left,
        x_right = c.x_right,
        tilt_deg = c.tilt_deg,
        clipped = c.clipped,
        confidence = c.confidence,
        candidate_count = count,
        frame_id = c.frame_id,
        timestamp_ms = c.timestamp_ms,
        distance_cm = _distance_cm(c, roi, ground, table),
        proximity = c.proximity,
    )


def compute_stop_line_distance(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
        config: StopLineDistanceConfig = StopLineDistanceConfig(),
        ground: GroundHomography | None = None,
        table: StopLineTable | None = None,
    ) -> tuple[StopLineResult, dict]:
    """(result, debug summary). ValueError if an input is None or the frames disagree."""
    check_same_frame(geometry, roi, "compute_stop_line_distance")
    log = []
    candidates = geometry.stop_line_candidates
    usable = []
    for c in candidates:
        if c.confidence < config.min_confidence:
            log.append(f"[REJECT] stop line at y={c.y_near_px:.1f} confidence "
                       f"{c.confidence:.3f} < {config.min_confidence}")
        else:
            usable.append(c)

    if usable:
        nearest = max(usable, key=lambda c: c.y_near_px)
        result = _measure(nearest, roi, ground, len(candidates), table)
        cm = "" if result.distance_cm is None else f" = {result.distance_cm:.1f}cm"
        log.append(f"[STOPLINE] {result.distance_px:.1f}px{cm} ahead"
                   f"{' (on it)' if result.clipped else ''}, tilt {result.tilt_deg:+.1f} deg")
    else:
        result = _nothing(len(candidates), geometry.frame_id, geometry.timestamp_ms)

    return result, {
        "frame_id": geometry.frame_id,
        "timestamp_ms": geometry.timestamp_ms,
        "candidate_count": len(candidates),
        "usable_count": len(usable),
        "log": log,
    }


def estimate_stop_line_distance(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
        config: StopLineDistanceConfig = StopLineDistanceConfig(),
        ground: GroundHomography | None = None,
        table: StopLineTable | None = None,
    ) -> StopLineResult:
    """Production twin of compute_stop_line_distance(), without the debug summary."""
    check_same_frame(geometry, roi, "estimate_stop_line_distance")
    candidates = geometry.stop_line_candidates
    nearest = None
    for c in candidates:
        if c.confidence >= config.min_confidence and (nearest is None or c.y_near_px > nearest.y_near_px):
            nearest = c
    if nearest is None:
        return _nothing(len(candidates), geometry.frame_id, geometry.timestamp_ms)
    return _measure(nearest, roi, ground, len(candidates), table)
