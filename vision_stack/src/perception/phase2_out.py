"""Phase 2 output packaging: one frame's results as the Phase 3 contract.

Purpose:
    The last Phase 2 stage and the contract boundary with Phase 3. It does no
    computation, filtering or decision logic: it collects fusion's detections
    and the lane offset, checks that they belong to the same frame, and
    packages them into one predictable object. Frame identity lives on the
    container, so a frame with no detections is still identifiable.

Main package:
    Phase2Output: the frame's detections exactly as fusion ordered them, its
    lane offset result, the frame identity and a detection count. Phase 3
    should steer lane-keeping from lane_offset_results, not from lane_boundary
    positions: those are bbox centroids, and for an angled marking that isn't
    where the marking meets the robot.

Flow:
    1. Take the detections and frame stamp from the FusionResult.
    2. Wrap the lane offset result in a list (empty if there is none).
    3. Check list types, required fields, per-item stamps and detection_count.
"""
from collections.abc import Callable
from dataclasses import dataclass

from src.perception.feature_fusion import DetectionObject, FusionResult
from src.perception.lane_offset import LaneOffsetResult

# The Phase 2 schema plus what Phase 3 needs to interpret a detection: which
# frame it is from and which ROI its position is local to
DETECTION_FIELDS = ("type", "position", "confidence", "timestamp", "frame_id",
                    "source_roi", "source_rect")
LANE_OFFSET_FIELDS = ("offset", "mode", "frame_id", "timestamp_ms")


@dataclass(frozen=True)
class Phase2Output:
    """
    Phase 2 -> Phase 3 handoff for one frame.

    Packaging rules:
        PR-1  detections and lane_offset_results are taken as-is: no
              re-ordering, no filtering.
        PR-2  detection_count is always len(detections).
        PR-3  frame_id and timestamp_ms come from the caller, never from a
              candidate or a clock.
        PR-4  Every detection and lane offset result carries the container's
              (frame_id, timestamp_ms).

    Failure cases, raised on construction:
        F1  detections or lane_offset_results is not a list (None included):
            TypeError. Pass [] for "nothing detected".
        F2  An item is missing a required field: AttributeError naming it.
            Only presence is checked; upstream stages own completeness.
        F3  detection_count is passed and isn't len(detections): ValueError.
        F4  frame_id or timestamp_ms not supplied: TypeError. There is no
            default, because a silent 0 would make Phase 3 treat every frame
            as the same instant.
        F5  An item's stamp differs from the container's: ValueError.
            Everything descends from one ROICropResult, so a mismatch means
            pieces from different frames were packaged together.
    """
    detections: list[DetectionObject]               # fusion order: traffic_light, lane_boundary by descending confidence, stop_sign
    lane_offset_results: list[LaneOffsetResult]     # one per frame from the chain (mode "none" when blind); [] if none supplied
    frame_id: int
    timestamp_ms: int
    detection_count: int | None = None              # None computes len(detections); a mismatch is rejected (F3)

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
            # The dataclass is frozen, so the computed count goes in via object.__setattr__
            object.__setattr__(self, "detection_count", n)
        elif self.detection_count != n:
            raise ValueError(
                f"Phase2Output: detection_count {self.detection_count} does "
                f"not match len(detections) {n}"
            )

def _check_items(
        kind: str,
        items: list,
        fields: tuple[str, ...],
        stamp: tuple[int, int],
        item_stamp: Callable[[object], tuple[int, int]],
    ) -> None:
    """Presence of every required field (F2), then agreement with the frame stamp (F5)."""
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


def package_phase2(
        fusion: FusionResult,
        lane_offset: LaneOffsetResult | None,
    ) -> Phase2Output:
    """
    Stage entry point: package one frame's fusion and lane offset results.

    Inputs:
        fusion: From fuse_detections(). An empty detections list is a valid frame.
        lane_offset: From compute_lane_offset(), or None if there is none for
            this frame.

    Outputs:
        Phase2Output stamped with fusion's frame_id and timestamp_ms, which
        fusion carried from the ROI crop.

    Raises:
        ValueError: If the lane offset result is from a different frame (F5).
    """
    return Phase2Output(
        detections = fusion.detections,
        lane_offset_results = [] if lane_offset is None else [lane_offset],
        frame_id = fusion.frame_id,
        timestamp_ms = fusion.timestamp_ms,
    )