"""
phase2_out.py

Phase 2 Output Packaging --- the last Phase 2 stage

Purpose:
    Collects what Phase 2 decided about one frame into a single Phase2Output
    and hands it to Phase 3. This stage performs no computation, filtering,
    thresholding or decision logic. It collects, checks that the pieces belong
    together, and packages.

    It is the contract boundary between Phase 2 and Phase 3, so it enforces:

    1. A fixed schema, so Phase 3 consumes one predictable object.
    2. One frame: every piece carries the frame the container names, so a
       package can never mix detections from one capture with a lane offset
       from another.
    3. Identity on empty frames: frame_id and timestamp_ms live on the
       container, so a frame with no detections is still identifiable.

Inputs (package_phase2):
    fusion: FusionResult from feature_fusion.fuse_detections()
        detections may be empty; an empty list is a valid frame.
    lane_offset: LaneOffsetResult from lane_offset.compute_lane_offset(), or
        None if there is none for this frame

Output (Phase2Output, frozen):
    .detections            list[DetectionObject], exactly as fusion produced
                           them: traffic_light, then lane_boundary
                           (descending confidence), then stop_sign. Not
                           re-sorted here
    .lane_offset_results   list[LaneOffsetResult]. The chain produces one per
                           frame (mode "none" when blind), so the list holds
                           one entry; an empty list means none was supplied
    .frame_id              int, from the capture loop
    .timestamp_ms          int, from the capture loop
    .detection_count       int, len(detections); computed when not passed

Coordinate space:
    DetectionObject.position and .bounding_box are ROI-LOCAL. Each detection
    names its ROI in source_roi and carries that ROI's frame rect in
    source_rect, so Phase 3 places it in the frame with:

        frame_x = det.position["x"] + det.source_rect[0]
        frame_y = det.position["y"] + det.source_rect[1]

    Steer lane-keeping from lane_offset_results, not from the position of a
    lane_boundary detection: a detection's position is its bounding-box
    centroid, which for an angled marking is not where the marking meets the
    robot. Lane offset anchors on the contour foot instead.

Packaging rules:
    PR-1  detections and lane_offset_results are taken as-is: no re-ordering,
          no filtering.
    PR-2  detection_count is always len(detections).
    PR-3  frame_id and timestamp_ms come from the caller (package_phase2 takes
          them from the FusionResult, which took them from the ROI crop), never
          from a candidate or a clock.
    PR-4  Every detection and lane offset result carries the container's
          (frame_id, timestamp_ms).

Failure cases:
    F1  detections or lane_offset_results is not a list (None included):
        TypeError. Pass an empty list explicitly for "nothing detected".
    F2  A detection or lane offset result is missing a required field:
        AttributeError naming the field. Upstream stages own field
        completeness; this stage only checks presence.
    F3  detection_count is passed and is not len(detections): ValueError.
    F4  frame_id or timestamp_ms not supplied: TypeError. They have no
        default: a silent 0 would make Phase 3 treat every frame as the same
        instant.
    F5  A detection or lane offset result carries a different stamp than the
        container: ValueError. Both come from one ROICropResult, so a
        disagreement means pieces from different frames were packaged
        together.
"""
from dataclasses import dataclass
from typing import List, Optional

from src.perception.feature_fusion import DetectionObject, FusionResult
from src.perception.lane_offset import LaneOffsetResult

# =============================================================================
# Required fields
# =============================================================================
# The Phase 2 schema plus what Phase 3 needs to interpret a detection: which
# frame it is from and which ROI its position is local to
DETECTION_FIELDS = ("type", "position", "confidence", "timestamp", "frame_id",
                    "source_roi", "source_rect")
LANE_OFFSET_FIELDS = ("offset", "mode", "frame_id", "timestamp_ms")

# =============================================================================
# Phase 2 Output
# =============================================================================
@dataclass(frozen=True)
class Phase2Output:
    """
    Phase 2 -> Phase 3 handoff for one frame

    detections: fused detections, as-is
    lane_offset_results: lane offset for this frame, as-is
    frame_id: frame identifier from the capture loop
    timestamp_ms: frame timestamp in ms from the capture loop
    detection_count: len(detections). Leave as None to have it computed; a
                     value that disagrees with len(detections) is rejected
    """
    detections: List[DetectionObject]
    lane_offset_results: List[LaneOffsetResult]
    frame_id: int
    timestamp_ms: int
    detection_count: Optional[int] = None

    def __post_init__(self):
        for name in ("detections", "lane_offset_results"):
            value = getattr(self, name)
            if not isinstance(value, list):
                raise TypeError(
                    f"Phase2Output: {name} must be a list, got "
                    f"{type(value).__name__}. Pass [] when there are none"
                )

        stamp = (self.frame_id, self.timestamp_ms)
        _check_items("detection", self.detections, DETECTION_FIELDS,
                     stamp, lambda d: (d.frame_id, d.timestamp))
        _check_items("lane offset result", self.lane_offset_results,
                     LANE_OFFSET_FIELDS, stamp,
                     lambda r: (r.frame_id, r.timestamp_ms))

        n = len(self.detections)
        if self.detection_count is None:
            object.__setattr__(self, "detection_count", n)
        elif self.detection_count != n:
            raise ValueError(
                f"Phase2Output: detection_count {self.detection_count} does "
                f"not match len(detections) {n}"
            )

def _check_items(kind, items, fields, stamp, item_stamp):
    """Presence of every required field, then agreement with the frame stamp."""
    for i, item in enumerate(items):
        for name in fields:
            if not hasattr(item, name):
                raise AttributeError(
                    f"Phase2Output: {kind} [{i}] is missing required field "
                    f"'{name}'"
                )
        if item_stamp(item) != stamp:
            raise ValueError(
                f"Phase2Output: {kind} [{i}] is stamped {item_stamp(item)} but "
                f"the frame is {stamp} --- pieces from different frames"
            )

# =============================================================================
# Stage entry point
# =============================================================================
def package_phase2(
        fusion: FusionResult,
        lane_offset: Optional[LaneOffsetResult],
    ) -> Phase2Output:
    """
    Purpose:
        Stage entry point, the same shape as the other stages: takes the
        previous stages' results and returns the next contract object

    Inputs:
        fusion: FusionResult from fuse_detections()
        lane_offset: LaneOffsetResult from compute_lane_offset(), or None

    Outputs:
        Phase2Output

    Notes:
        frame_id and timestamp_ms come from the FusionResult, which carried
        them from the ROI crop. Nothing is re-derived here
    """
    return Phase2Output(
        detections = fusion.detections,
        lane_offset_results = [] if lane_offset is None else [lane_offset],
        frame_id = fusion.frame_id,
        timestamp_ms = fusion.timestamp_ms,
    )