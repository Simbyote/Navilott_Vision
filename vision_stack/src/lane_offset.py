"""
lane_offset.py

Lane Offset Estimation

Purpose:
    Compute the lateral offset of the robot from the lane center using pixel positions of lane
    boundary candidates from feature fusion

Offset convention:
    0.0: robot is centered between detected boundaries
   -1.0: robot is at the left boundary
   +1.0: robot is at the right boundary

Notes:
    Output is normalized to frame width, making it resolution-independent
    and directly usable as a steering error signal in Phase 3 PID.
"""
from dataclasses import dataclass
from typing import List, Optional
from feature_fusion import DetectionObject

@dataclass
class LaneOffsetResult:
    """
    Results from lane offset computation for a single frame

    offset: normalized lateral offset from center
                    Negative = robot is right of lane center
                    Positive = robot is left of lane center
    left_x: pixel x of left boundary candidate
    right_x: pixel x of right boundary candidate
    lane_width_px: pixel distance between boundaries
    confidence: mean confidence of the two anchor candidates
    boundary_count: total lane_boundary detections this frame
    mode: "two_boundary" | "left_only" | "right_only" | "none"
    frame_id: frame identifier from capture loop
    timestamp: timestamp from capture loop
    """
    offset: float
    left_x: Optional[float]
    right_x: Optional[float]
    lane_width_px: Optional[float]
    confidence: float
    boundary_count: int
    mode: str
    frame_id: int
    timestamp: int

def compute_lane_offset(
    detections:       List[DetectionObject],
    frame_width:      int,
    frame_id:         int,
    timestamp:        int,
    conf_threshold:   float = 0.30,
    min_lane_width_px: float = 150.0,
) -> LaneOffsetResult:
    """
    Purpose:
        Compute lateral offset from lane center using fusion output positions.

    Inputs:
        detections: list[DetectionObject] from fuse_detections()
        frame_width: width of the lane ROI in pixels
        frame_id: from capture loop
        timestamp: timestamp from capture loop
        conf_threshold: minimum confidence to use a candidate as a boundary anchor
        min_lane_width_px: minimum pixel distance between lane boundaries

    Output:
        LaneOffsetResult  : final lane offset estimate
    """
    frame_center = frame_width / 2.0

    # Pull lane boundary detections above confidence threshold
    lanes = [
        d for d in detections
        if d.type == "lane_boundary" and d.confidence >= conf_threshold
    ]

    boundary_count = len(lanes)

    # No lane boundaries detected = no lane offset returned
    if not lanes:
        return LaneOffsetResult(
            offset=0.0, left_x=None, right_x=None,
            lane_width_px=None, confidence=0.0,
            boundary_count=0, mode="none",
            frame_id=frame_id, timestamp=timestamp,
        )

    # Sort candidates by x position
    lanes_by_x = sorted(lanes, key=lambda d: d.position["x"])

    left  = lanes_by_x[0]
    right = lanes_by_x[-1]

    left_x  = left.position["x"]
    right_x = right.position["x"]

    # =========================================================================
    # One boundary detection
    # =========================================================================
    if left is right:
        if left_x < frame_center:   # Detected boundary is on the left
            offset = (frame_center - left_x) / frame_center
            mode   = "left_only"
        else:                       # Detected boundary is on the right
            offset = (frame_center - right_x) / frame_center
            mode   = "right_only"

        # Return only the applicable boundary anchor
        return LaneOffsetResult(
            offset=round(offset, 4),
            left_x=left_x if mode == "left_only" else None,
            right_x=right_x if mode == "right_only" else None,
            lane_width_px=None,
            confidence=round(left.confidence, 4),
            boundary_count=boundary_count,
            mode=mode,
            frame_id=frame_id,
            timestamp=timestamp,
        )

    # ==========================================================================
    # Two boundary detections - checking width
    # ==========================================================================
    lane_center    = (left_x + right_x) / 2.0
    lane_width_px  = right_x - left_x

    # Ensure the lane detected is as wide as expected
    if lane_width_px < min_lane_width_px:
        mid_x  = (left_x + right_x) / 2.0
        offset = (frame_center - mid_x) / frame_center
        offset = max(-1.0, min(1.0, offset))

        # Return the average of the two anchors
        return LaneOffsetResult(
            offset=round(offset, 4),
            left_x=None, right_x=None,
            lane_width_px=None,
            confidence=round((left.confidence + right.confidence) / 2.0, 4),
            boundary_count=boundary_count,
            mode="width_rejected",
            frame_id=frame_id, timestamp=timestamp,
        )

    # ==========================================================================
    # Two boundary detections - computing offset
    # ======================================================t====================
    lane_center = (left_x + right_x) / 2.0
    offset      = (lane_center - frame_center) / (lane_width_px / 2.0)
    offset      = max(-1.0, min(1.0, offset))
    mean_conf   = (left.confidence + right.confidence) / 2.0

    # Return the average of the two anchors
    return LaneOffsetResult(
        offset=round(offset, 4),
        left_x=left_x,
        right_x=right_x,
        lane_width_px=round(lane_width_px, 2),
        confidence=round(mean_conf, 4),
        boundary_count=boundary_count, 
        mode="two_boundary",
        frame_id=frame_id, timestamp=timestamp,
    )