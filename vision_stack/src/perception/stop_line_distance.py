"""Stop-line distance: how far ahead of the robot the nearest stop line is.

Purpose:
    Turns geometry's stop-line candidates into one measurement per frame:
    the rows between the nearest stop line and the bottom of the lane ROI,
    the nearest ground the camera sees. Geometry only detects; this module
    measures, the way lane_offset measures lateral error from the same
    geometry result. The two are siblings over one frame:

        crop_rois -> geometry -+-> compute_lane_offset       -> lateral error
                               +-> compute_stop_line_distance -> distance ahead

    The distance is in lane-ROI px. Converting it to cm needs a ground
    homography: cm_per_px is only valid at the bottom row, and perspective
    compresses the rows above it.

Main package:
    StopLineResult: whether a stop line was found, its distance_px from the
    bottom of the lane ROI (0 = the robot is on it), its ends, tilt and
    confidence, and the frame identity. One per frame; detected False when
    there is none.

Flow:
    1. Gate each candidate by confidence.
    2. Keep the one nearest the robot (largest y_near_px).
    3. distance_px = lane ROI height - y_near_px.
"""
from dataclasses import dataclass

from src.perception.geometry import GeometryBranchResult, StopLineCandidate
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
    the lane ROI.
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


# =============================================================================
# Measurement
# =============================================================================

def _nothing(count: int, frame_id: int, timestamp_ms: int) -> StopLineResult:
    return StopLineResult(
        detected = False, distance_px = None, y_near_px = None, x_left = None,
        x_right = None, tilt_deg = None, clipped = False, confidence = 0.0,
        candidate_count = count, frame_id = frame_id, timestamp_ms = timestamp_ms,
    )

def _measure(
        c: StopLineCandidate,
        roi_h: int,
        count: int,
    ) -> StopLineResult:
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
    )

def compute_stop_line_distance(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
        config: StopLineDistanceConfig = StopLineDistanceConfig(),
    ) -> tuple[StopLineResult, dict]:
    """
    Measure the distance to the nearest confident stop line.

    Inputs:
        geometry: From run_geometry_stage().
        roi: Supplies the lane ROI height (the reference row) and the frame stamp.

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
        result = _measure(nearest, roi.lane_rect[3], len(candidates))
        log.append(f"[STOPLINE] {result.distance_px:.1f}px ahead"
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
    return _measure(nearest, roi.lane_rect[3], len(candidates))
