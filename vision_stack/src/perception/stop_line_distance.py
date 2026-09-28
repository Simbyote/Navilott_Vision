"""Stop-line distance: how far ahead of the robot the nearest stop line is.

Purpose:
    Turns geometry's stop-line candidates into one measurement per frame.
    Geometry only detects; this module measures, the way lane_offset
    measures lateral error from the same geometry result. The two are
    siblings over one frame:

        crop_rois -> geometry -+-> compute_lane_offset       -> lateral error
                               +-> compute_stop_line_distance -> distance ahead

    Two distances come out. distance_px counts lane-ROI rows from the line's
    nearest point to the ROI bottom, and needs no calibration. distance_cm is
    the floor distance forward from the robot reference point to where the
    line's near edge crosses the robot's centerline (X = 0), through the
    ground homography (perception/ground.py); None when there is none. The
    reference point is the floor at the bottom of the camera's view, so a
    line the robot is on measures 0 in both.

Main package:
    StopLineResult: whether a stop line was found, its distance_px and
    distance_cm, how close it is as a [0, 1] proximity, its ends, tilt and
    confidence, and the frame identity. One per frame; detected False when
    there is none.

Flow:
    1. Gate each candidate by confidence.
    2. Keep the one nearest the robot (largest y_near_px).
    3. distance_px = lane ROI height - y_near_px.
    4. distance_cm: the near edge's two ends to frame px (add the lane ROI
       origin), to the floor, then the crossing with X = 0.
"""
import math
from dataclasses import dataclass

from src.perception.geometry import GeometryBranchResult, StopLineCandidate
from src.perception.ground import GroundHomography
from src.perception.roi_crop import ROICropResult
from src.utils import check_same_frame


# =============================================================================
# Config and result
# =============================================================================

@dataclass(frozen=True)
class StopLineDistanceConfig:
    """Tuning for the stop-line measurement."""
    min_confidence: float = 0.4     # candidates below this aren't measured


@dataclass(frozen=True)
class StopLineResult:
    """
    The nearest stop line in one frame. frame_id and timestamp_ms are carried
    from capture, never re-derived.

    Rows grow downward, toward the robot; the reference row is the bottom of
    the lane ROI, which is the bottom of the frame and of the camera's view.
    """
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
    # Floor cm forward of the robot reference to where the near edge crosses
    # the robot's centerline; 0 when on it. None without a ground homography
    distance_cm: float | None = None
    proximity: float = 0.0          # [0, 1] y_near_px / ROI height; 1.0 = at the ROI bottom


# =============================================================================
# Measurement
# =============================================================================

def _nothing(count: int, frame_id: int, timestamp_ms: int) -> StopLineResult:
    return StopLineResult(
        detected = False, distance_px = None, y_near_px = None, x_left = None,
        x_right = None, tilt_deg = None, clipped = False, confidence = 0.0,
        candidate_count = count, frame_id = frame_id, timestamp_ms = timestamp_ms,
        distance_cm = None, proximity = 0.0,
    )

def _distance_cm(
        c: StopLineCandidate,
        roi: ROICropResult,
        ground: GroundHomography | None,
    ) -> float | None:
    """
    Floor distance to where the candidate's near edge crosses the robot's centerline.

    The near (bottom) edge is rebuilt at both ends from its midpoint row and
    tilt; for a clipped line it is the ROI bottom, where the robot already
    is. Lane-ROI coordinates become frame coordinates by adding the ROI
    origin before projecting. None without a ground plane, or when the frame
    isn't the size the ground plane was fit at.
    """
    if ground is None or tuple(roi.source_shape[:2]) != (ground.image_size[1], ground.image_size[0]):
        return None
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
        distance_cm = _distance_cm(c, roi, ground),
        proximity = c.proximity,
    )

def compute_stop_line_distance(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
        config: StopLineDistanceConfig = StopLineDistanceConfig(),
        ground: GroundHomography | None = None,
    ) -> tuple[StopLineResult, dict]:
    """
    Measure the distance to the nearest confident stop line.

    Inputs:
        geometry: From run_geometry_stage().
        roi: Supplies the lane ROI's origin and height (the reference row),
            the frame size and the frame stamp.
        ground: The ground homography (PipelineConfig.ground); None leaves
            distance_cm None.

    Outputs:
        (result, debug_summary). debug_summary holds frame_id, timestamp_ms,
        candidate_count, usable_count and log.

    Raises:
        ValueError: If either input is None, or their frame stamps disagree.
    """
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
        result = _measure(nearest, roi, ground, len(candidates))
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
    ) -> StopLineResult:
    """
    Production twin of compute_stop_line_distance(): same measurement, no debug summary or log.

    Raises:
        ValueError: If either input is None, or their frame stamps disagree.
    """
    check_same_frame(geometry, roi, "estimate_stop_line_distance")
    candidates = geometry.stop_line_candidates
    nearest = None
    for c in candidates:
        if c.confidence >= config.min_confidence and (nearest is None or c.y_near_px > nearest.y_near_px):
            nearest = c
    if nearest is None:
        return _nothing(len(candidates), geometry.frame_id, geometry.timestamp_ms)
    return _measure(nearest, roi, ground, len(candidates))
