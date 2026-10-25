"""Phase 2 output packaging: one frame's results as the Phase 3 contract.

Purpose:
    The boundary with Phase 3. It computes nothing: it collects one frame's
    detections and measurements, checks they all belong to that frame, and
    packages them, so even a frame with no detections is identifiable.

Main package:
    Phase2Output. Phase 3 steers from lane_offset_results, not from the
    lane_boundary centroids, which for an angled line aren't where it meets
    the robot.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

from src.perception.feature_fusion import DetectionObject, FusionResult
from src.perception.lane_offset import LaneOffsetResult
from src.perception.obstacle import ObstacleResult
from src.perception.stop_line_distance import StopLineResult

# Plus the frame and the ROI a detection's position is local to
DETECTION_FIELDS = ("type", "position", "confidence", "timestamp", "frame_id",
                    "source_roi", "source_rect")
LANE_OFFSET_FIELDS = ("offset", "mode", "frame_id", "timestamp_ms")
STOP_LINE_FIELDS = ("detected", "distance_px", "frame_id", "timestamp_ms")
OBSTACLE_FIELDS = ("detected", "distance_px", "distance_cm", "clipped", "frame_id", "timestamp_ms")


@dataclass(frozen=True)
class Phase2Output:
    """
    Lists are taken as-is, never reordered or filtered, and every item carries
    the container's frame_id and timestamp_ms.

    Raised on construction: TypeError if a list is None or a stamp is missing (a
    silent 0 would make every frame the same instant); AttributeError for a
    missing field; ValueError if detection_count is wrong or an item's stamp
    differs, which means two frames were packaged together.
    """
    detections: list[DetectionObject]               # fusion order: traffic_light, lane_boundary by descending confidence, stop_sign
    lane_offset_results: list[LaneOffsetResult]     # one per frame from the chain (mode "none" when blind); [] if none supplied
    frame_id: int
    timestamp_ms: int
    detection_count: int | None = None              # None computes len(detections); a mismatch is rejected (F3)
    stop_line_results: list[StopLineResult] = field(default_factory=list)
    obstacle_results: list[ObstacleResult] = field(default_factory=list)

    def __post_init__(self):
        for name in ("detections", "lane_offset_results", "stop_line_results", "obstacle_results"):
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
        _check_items("stop line result", self.stop_line_results,
                     STOP_LINE_FIELDS, stamp,
                     lambda r: (r.frame_id, r.timestamp_ms))
        _check_items("obstacle result", self.obstacle_results,
                     OBSTACLE_FIELDS, stamp,
                     lambda r: (r.frame_id, r.timestamp_ms))

        n = len(self.detections)
        if self.detection_count is None:
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
        stop_line: StopLineResult | None = None,
        obstacle: ObstacleResult | None = None,
    ) -> Phase2Output:
    """Stamped with fusion's frame. A None measurement becomes []; ValueError if one is from another frame."""
    return Phase2Output(
        detections = fusion.detections,
        lane_offset_results = [] if lane_offset is None else [lane_offset],
        stop_line_results = [] if stop_line is None else [stop_line],
        obstacle_results = [] if obstacle is None else [obstacle],
        frame_id = fusion.frame_id,
        timestamp_ms = fusion.timestamp_ms,
    )