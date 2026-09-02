"""
lane_offset.py

Lane Offset Estimation

Purpose:
    Reduce the lane_boundary detections for one frame to a SINGLE signed
    lateral offset. This is the only place in the codebase that performs this
    reduction. Phase 3 consumes the result; it must not re-derive it.

Sign convention (canonical, see contracts.py):
    offset > 0  ->  lane centre appears right of image centre
                ->  robot is LEFT of lane centre, must steer RIGHT
    offset < 0  ->  robot is RIGHT of lane centre, must steer LEFT
    offset = 0  ->  centred

    Magnitude is normalised to one lane half-width, so +1.0 means the robot is
    riding a boundary. Output is resolution-independent.

CHANGES FROM THE PREVIOUS REVISION
    1. Sign convention unified. The one-anchor and width-rejected branches used
       (frame_center - x), the inverse of the two-anchor branch. 195 frames of
       session2.log (6.8%) ran through an inverted branch -- and those were
       exactly the frames where a lane is partially lost, i.e. intersections
       and wall-adjacent starts. All branches now use (x - frame_center).
    2. Anchors are selected by confidence within each side of frame centre, as
       phase2_pipeline.md always documented. The old code took lanes_by_x[0]
       and lanes_by_x[-1] -- leftmost and rightmost, ignoring confidence -- so
       any spurious edge anywhere in the frame became an anchor by position
       alone. That is what let a wall seam become the left anchor.
    3. Added max_lane_width_px. A pair spanning most of the frame is not a
       lane; it is one real boundary plus one piece of scenery. This is the
       wall signature and is now rejected at the estimator, independently of
       any geometry.py re-calibration.
    4. An out-of-band pair no longer emits the midpoint of a pair we have just
       declared implausible. It falls back to the higher-confidence single
       anchor, which is the only self-consistent choice.
    5. Single-anchor offset is inferred against nominal_lane_width_px instead
       of frame_center, so its scale matches the two-anchor branch.
"""
from dataclasses import dataclass
from typing import List, Optional

from feature_fusion import DetectionObject
from contracts import (
    LaneEstimate,
    LANE_MODE_TWO_BOUNDARY, LANE_MODE_LEFT_ONLY, LANE_MODE_RIGHT_ONLY,
    LANE_MODE_WIDTH_REJECTED, LANE_MODE_NONE,
)


@dataclass
class LaneOffsetResult:
    """
    Results from lane offset computation for a single frame.

    offset: normalised lateral offset, [-1.0, +1.0]
            Positive = robot is LEFT of lane centre (must steer right)
            Negative = robot is RIGHT of lane centre (must steer left)
    left_x: pixel x of the left boundary anchor; None if not used
    right_x: pixel x of the right boundary anchor; None if not used
    lane_width_px: pixel distance between anchors; None unless two were used
    confidence: mean confidence of the anchors actually used
    boundary_count: total lane_boundary detections above the confidence gate
    mode: "two_boundary" | "left_only" | "right_only" | "width_rejected" | "none"
    frame_id: frame identifier from the capture loop
    timestamp: timestamp from the capture loop
    """
    offset:         float
    left_x:         Optional[float]
    right_x:        Optional[float]
    lane_width_px:  Optional[float]
    confidence:     float
    boundary_count: int
    mode:           str
    frame_id:       int
    timestamp:      int

    def to_lane_estimate(self) -> LaneEstimate:
        """
        Purpose:
            Convert to the seam contract consumed by Phase 3.
            This is the ONLY path by which a lane offset may reach Phase 3.
        """
        return LaneEstimate(
            offset_norm=self.offset,
            mode=self.mode,
            confidence=self.confidence,
            valid=(self.mode != LANE_MODE_NONE),
            boundary_count=self.boundary_count,
            lane_width_px=self.lane_width_px,
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp,
        )


def _clamp_unit(value: float) -> float:
    """
    Purpose:
        Clamp to the normalised offset range [-1.0, +1.0].
    """
    return max(-1.0, min(1.0, value))


def _single_anchor_offset(
        anchor_x: float,
        is_left: bool,
        frame_center: float,
        nominal_half_width: float,
    ) -> float:
    """
    Purpose:
        Infer the lane-centre offset from ONE boundary anchor.

    Inputs:
        anchor_x: x position of the anchor, ROI pixels
        is_left: True if the anchor is the left boundary of the lane
        frame_center: x of the image centre column
        nominal_half_width: assumed lane half-width in px (TODO-CALIBRATE)

    Outputs:
        offset: normalised, canonical sign convention

    Notes:
        Seeing only the left boundary means the lane centre sits one half-width
        to its right, and vice versa. The previous implementation divided by
        frame_center, which put the single-anchor branch on a different scale
        AND a different polarity from the two-anchor branch -- so the offset
        magnitude jumped whenever a boundary dropped out, which is exactly what
        happens on the reacquisition frame after an intersection.
    """
    lane_center = anchor_x + nominal_half_width if is_left else anchor_x - nominal_half_width
    return _clamp_unit((lane_center - frame_center) / nominal_half_width)


def compute_lane_offset(
    detections: List[DetectionObject],
    frame_width: int,
    frame_id: int,
    timestamp: int,
    conf_threshold: float = 0.30,
    min_lane_width_px: float = 150.0,
    max_lane_width_px: Optional[float] = None,
    nominal_lane_width_px: float = 240.0,
) -> LaneOffsetResult:
    """
    Purpose:
        Compute the signed lateral offset from lane centre for one frame.

    Inputs:
        detections: list[DetectionObject] from fuse_detections()
        frame_width: width of the lane ROI in pixels
        frame_id: from the capture loop
        timestamp: from the capture loop
        conf_threshold: minimum confidence for a candidate to act as an anchor
        min_lane_width_px: anchor separation below this is implausible
        max_lane_width_px: anchor separation above this is implausible.
            Defaults to 0.80 * frame_width. This is the wall guard: a real lane
            does not span four fifths of the frame.
        nominal_lane_width_px: assumed lane width for single-anchor inference
            TODO-CALIBRATE against trackT3/T4/T5

    Output:
        LaneOffsetResult
    """
    frame_center = frame_width / 2.0
    nominal_half = max(nominal_lane_width_px / 2.0, 1.0)
    if max_lane_width_px is None:
        max_lane_width_px = 0.80 * frame_width

    lanes = [
        d for d in detections
        if d.type == "lane_boundary" and d.confidence >= conf_threshold
    ]
    boundary_count = len(lanes)

    if not lanes:
        return LaneOffsetResult(
            offset=0.0, left_x=None, right_x=None, lane_width_px=None,
            confidence=0.0, boundary_count=0, mode=LANE_MODE_NONE,
            frame_id=frame_id, timestamp=timestamp,
        )

    # =========================================================================
    # Anchor selection: best PAIR, scored by confidence and width plausibility.
    #
    # Do NOT partition candidates by which side of frame centre they fall on.
    # Both boundaries of the lane sit on the same side of the image whenever
    # the robot is meaningfully off-centre, which is exactly the situation the
    # estimator exists to detect. "Left" and "right" are defined relative to
    # EACH OTHER, not to the image.
    #
    # Scoring a pair on confidence alone is not enough either: a high-confidence
    # wall seam pairs with a real lane line and wins on confidence. A lane has a
    # roughly known width, so width plausibility is the discriminator that
    # separates a lane pair from a lane-plus-scenery pair.
    #
    # Cost is O(N^2) over the candidates that cleared the confidence gate.
    # N is a handful of contours per frame, so this is a few dozen float
    # comparisons -- negligible against the ~50 ms/frame budget.
    # =========================================================================
    best_pair = None
    best_score = -1.0
    for i, a in enumerate(lanes):
        for b in lanes[i + 1:]:
            lo, hi = (a, b) if a.position["x"] <= b.position["x"] else (b, a)
            width = hi.position["x"] - lo.position["x"]
            if width < min_lane_width_px or width > max_lane_width_px:
                continue
            # 1.0 when the pair is exactly nominal width, falling to 0.0 as it
            # departs by a full nominal width.
            width_plausibility = max(
                0.0, 1.0 - abs(width - nominal_lane_width_px) / max(nominal_lane_width_px, 1.0))
            score = ((lo.confidence + hi.confidence) / 2.0) * width_plausibility
            if score > best_score:
                best_score, best_pair = score, (lo, hi)

    # =========================================================================
    # Fallback: no plausible pair. Infer from the single anchor we trust most.
    # Never emit the midpoint of a pair we have just declared implausible.
    # =========================================================================
    if best_pair is None:
        anchor = max(lanes, key=lambda d: d.confidence)
        is_left = anchor.position["x"] < frame_center
        offset = _single_anchor_offset(
            anchor.position["x"], is_left, frame_center, nominal_half)

        if boundary_count == 1:
            mode = LANE_MODE_LEFT_ONLY if is_left else LANE_MODE_RIGHT_ONLY
            conf = anchor.confidence
        else:
            mode = LANE_MODE_WIDTH_REJECTED
            conf = anchor.confidence * 0.5   # penalise: a pair existed and failed

        return LaneOffsetResult(
            offset=round(offset, 4),
            left_x=anchor.position["x"] if is_left else None,
            right_x=None if is_left else anchor.position["x"],
            lane_width_px=None,
            confidence=round(conf, 4),
            boundary_count=boundary_count,
            mode=mode,
            frame_id=frame_id, timestamp=timestamp,
        )

    # =========================================================================
    # Two anchors, plausible separation
    # =========================================================================
    left, right = best_pair
    left_x = left.position["x"]
    right_x = right.position["x"]
    lane_width_px = right_x - left_x

    lane_center = (left_x + right_x) / 2.0
    offset = _clamp_unit((lane_center - frame_center) / (lane_width_px / 2.0))
    mean_conf = (left.confidence + right.confidence) / 2.0

    return LaneOffsetResult(
        offset=round(offset, 4),
        left_x=left_x,
        right_x=right_x,
        lane_width_px=round(lane_width_px, 2),
        confidence=round(mean_conf, 4),
        boundary_count=boundary_count,
        mode=LANE_MODE_TWO_BOUNDARY,
        frame_id=frame_id, timestamp=timestamp,
    )