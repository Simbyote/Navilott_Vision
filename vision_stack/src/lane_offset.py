"""
lane_offset.py

Lane Offset Estimation

Purpose:
    Compute the lateral offset of the robot from the lane center using pixel positions of lane
    boundary candidates from feature fusion

Offsets:
    0.0: robot is centered between detected boundaries
   -1.0: robot is at the left boundary
   +1.0: robot is at the right boundary

Notes:
    Output is normalized to frame width, making it resolution-independent
    and directly usable as a steering error signal in Phase 3 PID

@TODO: Does not have its own Standalone Test Block
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
    detections: List[DetectionObject],
    frame_width: int,
    frame_id: int,
    timestamp: int,
    conf_threshold: float = 0.30,
    min_lane_width_px: float = 150.0,
    fix_cfg = None,
    tags = None,
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
        fix_cfg: Optional[Config] from the config.py
        NOTE: anchor_halves=True enables RESOLUTION 5
            The left anchor lies in the left haf of the ROI and the right anchor
            lies on the right half of the ROI. WHen all candidates fall one to one
            half, the frame downgrades to left only or right only instead of two 
            boundaries
        tags: Optional[FrameTags] from the config.py
        NOTE: anchor_wrong_half is set when a two-anchor result used anchors from the
            same ROI half

    Output:
        LaneOffsetResult : final lane offset estimate
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

    # =========================================================================
    # Anchor selection
    # Extremes of the sorted list, regardless of ROI half
    # RESOLUTION 5 pools candidates per ROI half; left anchor is the
    # leftmost of the left pool, right anchor the rightmost of the right pool
    # An empty pool downgrades the frame to single-boundary mode
    # =========================================================================
    fix5_on = fix_cfg is not None and fix_cfg.anchor_halves
    if fix5_on:
        left_pool  = [d for d in lanes_by_x if d.position["x"] <  frame_center]
        right_pool = [d for d in lanes_by_x if d.position["x"] >= frame_center]
        left  = left_pool[0]   if left_pool  else None
        right = right_pool[-1] if right_pool else None
    else:
        left = lanes_by_x[0]
        right = lanes_by_x[-1]

    # =========================================================================
    # One boundary detection
    # =========================================================================
    if left is None or right is None or left is right:
        anchor = left if left is not None else right
        anchor_x = anchor.position["x"]

        offset = (frame_center - anchor_x) / frame_center
        mode = "left_only" if anchor_x < frame_center else "right_only"

        # Return only the applicable boundary anchor
        return LaneOffsetResult(
            offset=round(offset, 4),
            left_x=anchor_x if mode == "left_only" else None,
            right_x=anchor_x if mode == "right_only" else None,
            lane_width_px=None,
            confidence=round(anchor.confidence, 4),
            boundary_count=boundary_count,
            mode=mode,
            frame_id=frame_id,
            timestamp=timestamp,
        )
    
    left_x = left.position["x"]
    right_x = right.position["x"]

    # INSTRUMENTATION: two-anchor result built from anchors in the same ROI
    # half
    if tags is not None and (left_x < frame_center) == (right_x < frame_center):
        tags.anchor_wrong_half = True

    # ==========================================================================
    # Two boundary detections - checking width
    # ==========================================================================
    lane_center = (left_x + right_x) / 2.0
    lane_width_px = right_x - left_x

    # Ensure the lane detected is as wide as expected
    if lane_width_px < min_lane_width_px:
        mid_x = (left_x + right_x) / 2.0
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
    offset = (lane_center - frame_center) / (lane_width_px / 2.0)
    offset = max(-1.0, min(1.0, offset))
    mean_conf = (left.confidence + right.confidence) / 2.0

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