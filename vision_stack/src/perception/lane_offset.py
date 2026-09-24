"""Lane offset: the robot's lateral error from the center of its own lane.

Purpose:
    Turns the geometry branch's lane candidates into a steering error. It
    reads GeometryBranchResult rather than fusion output, because fusion
    keeps only a centroid and a confidence, and estimation needs the geometry
    that normalization discards. The two are siblings over the same frame:

        crop_rois -> geometry -+-> fuse_detections     -> Phase2Output
                               +-> compute_lane_offset -> steering error

Main package:
    LaneOffsetResult: offset in [-1, 1], normalized by half the lane ROI
    width (0 = centered, + = robot right of lane center, - = left), the
    boundary anchors used, a confidence, and the mode the estimate came from.
    Usable directly as the Phase 3 steering error.

Flow:
    1. Gate each lane candidate: is it trustworthy enough to steer by?
    2. Reduce survivors to anchors: foot x and a proximity-scaled weight.
    3. Pick the nearest boundary on each side of ROI center.
    4. Two boundaries with plausible spacing: offset from their midpoint.
    5. Otherwise one boundary: project the lane center from it, if calibrated.
"""
import numpy as np
from dataclasses import dataclass

from src.params import (
    FOOT_BAND_PX, MODE_LEFT_ONLY, MODE_NONE, MODE_RIGHT_ONLY,
    MODE_SINGLE_UNCALIBRATED, MODE_TWO_BOUNDARY,
)
from src.utils import check_same_frame, clamp
from src.perception.geometry import GeometryBranchResult, LaneCandidate
from src.perception.roi_crop import ROICropResult


@dataclass(frozen=True)
class LaneOffsetConfig:
    """
    Tuning for lane offset estimation.

    These gates are a second, stricter pass than LaneContourFilter. Geometry
    decides "is this a lane marking"; this decides "is it trustworthy enough
    to steer by". A candidate can legitimately pass the first and fail this.
    """
    conf_threshold: float = 0.30        # min candidate confidence
    min_proximity: float = 0.25         # [0, 1]; below this the candidate is too far up the ROI to describe where the robot is now
    min_length_px: float = 25.0         # minAreaRect long side; keeps dashed-center-line fragments from anchoring a boundary
    min_width_px: float = 1.0           # minAreaRect short side; below this is noise
    max_width_px: float = 25.0          # above this it's a blob, glare patch or merged pair, not one painted line
    min_intensity: float = 90.0         # 0-255 mean inside the contour; rejects shadow edges and mat seams
    min_lane_width_px: float = 60.0     # anchors closer than this are fragments of one marking
    max_lane_width_px: float = 400.0    # anchors wider apart than this span more than one lane
    # Distance from a boundary to the lane center, lane-ROI px. Belongs to
    # camera calibration (cm-per-px against a known lane width). None disables
    # single-boundary mode: one usable boundary then reports
    # "single_uncalibrated" with zero confidence, not a steering error of unknown scale.
    expected_half_lane_px: float | None = 228        # Half of 95% of a single lane frame
    foot_band_px: int = FOOT_BAND_PX    # band height for foot_x; only used when geometry didn't store one


@dataclass(frozen=True)
class BoundaryAnchor:
    """One lane boundary reduced to the quantities the offset math uses."""
    foot_x: float               # lane-ROI px, from the contour's lowest rows
    weight: float               # confidence scaled by proximity; how much this anchor counts
    candidate: LaneCandidate    # kept for debug, and for the scene state machine to reach back into

@dataclass(frozen=True)
class LaneOffsetResult:
    """
    Lane offset estimate for one frame. frame_id and timestamp_ms are carried
    from capture, never re-derived.

    boundary_count counts usable boundaries, not raw detections. A frame with
    four contours and none usable reports 0, which separates "saw nothing"
    from "saw nothing it could steer by".
    """
    offset: float                   # [-1, 1] of half the lane ROI width; + = robot right of lane center
    left_x: float | None            # left anchor x, lane-ROI px; None if unused
    right_x: float | None           # right anchor x, lane-ROI px; None if unused
    lane_width_px: float | None     # right_x - left_x; None in single-sided modes
    confidence: float               # mean weight of the anchors actually used
    boundary_count: int
    mode: str                       # "two_boundary" | "left_only" | "right_only" | "single_uncalibrated" | "none"
    frame_id: int
    timestamp_ms: int


def foot_x(
        candidate: LaneCandidate,
        band_px: int = FOOT_BAND_PX
    ) -> float:
    """
    Anchor x for a boundary: where the marking sits at its closest approach to the robot.

    Purpose:
        A bbox centroid puts an angled line's anchor halfway up the ROI,
        which isn't where the robot is about to arrive. A near-vertical
        marking gives the same answer either way.

    Inputs:
        band_px: Rows above the contour's lowest point averaged in the
            contour fallback. Unused when the candidate carries a foot_x.

    Outputs:
        Lane-ROI px, in order of preference: the candidate's stored foot_x
        (any value >= 0), the mean x of the contour's lowest band, or the
        bbox horizontal center when there's no contour (hand-built candidates).
    """
    # Geometry computed foot_x from the full contour. Preferring it also covers
    # CSV replay, where candidates arrive without a contour to recompute from.
    stored = getattr(candidate, "foot_x", -1.0)
    if stored is not None and stored >= 0.0:
        return float(stored)

    contour = candidate.contour
    if contour is None or len(contour) == 0:
        x, _, w, _ = candidate.bbox
        return float(x) + float(w) / 2.0

    pts = np.asarray(contour).reshape(-1, 2)
    y_max = pts[:, 1].max()
    band = pts[pts[:, 1] >= y_max - max(band_px, 0)]
    return float(band[:, 0].mean())

def _usable(
        candidate: LaneCandidate,
        config: LaneOffsetConfig,
        log: list[str],
    ) -> bool:
    """
    Whether a candidate is trustworthy enough to anchor a boundary.

    Logs the gate that rejected it, so the debug summary answers "why did
    this frame go blind" without a rerun.
    """
    c = candidate
    if c.confidence < config.conf_threshold:
        log.append(f"[REJECT] confidence {c.confidence:.3f} < {config.conf_threshold}")
        return False
    if c.proximity < config.min_proximity:
        log.append(f"[REJECT] proximity {c.proximity:.3f} < {config.min_proximity} "
                   "(too far up the ROI)")
        return False
    if c.length_px < config.min_length_px:
        log.append(f"[REJECT] length_px {c.length_px:.1f} < {config.min_length_px} "
                   "(fragment)")
        return False
    if not (config.min_width_px <= c.width_px <= config.max_width_px):
        log.append(f"[REJECT] width_px {c.width_px:.1f} outside "
                   f"[{config.min_width_px}, {config.max_width_px}]")
        return False
    if c.mean_intensity < config.min_intensity:
        log.append(f"[REJECT] mean_intensity {c.mean_intensity:.1f} < "
                   f"{config.min_intensity} (too dim to be tape)")
        return False
    return True

def _anchor(
        candidate: LaneCandidate,
        config: LaneOffsetConfig,
    ) -> BoundaryAnchor:
    """Reduce a usable candidate to its anchor x and steering weight."""
    # Proximity scales confidence rather than replacing it: a strong marking at
    # the top of the ROI is real, it just says less about where the robot is now
    weight = candidate.confidence * (0.5 + 0.5 * candidate.proximity)
    return BoundaryAnchor(
        foot_x = foot_x(candidate, config.foot_band_px),
        weight = round(weight, 4),
        candidate = candidate,
    )

def _lane_pair(
        anchors: list[BoundaryAnchor],
        center_x: float,
    ) -> tuple[BoundaryAnchor | None, BoundaryAnchor | None]:
    """
    Select the two boundaries of the lane nearest the robot.

    Purpose:
        The robot sits at ROI center, so its lane is bounded by the nearest
        usable boundary on each side, not the outermost pair. On a two-lane
        street with a visible divider, the outermost pair centers the robot
        on the street: an error that always points the same way.

    Rules:
        1. Anchors on both sides of center: take the nearest on each side.
           They bracket the robot, so they are its own lane.
        2. Every anchor on one side: the robot has drifted out of its lane.
           The two nearest center still describe the nearest lane, and the
           offset is the error that steers back. Requiring a straddling pair
           would drop a usable measurement on exactly the frames where it
           matters most.
        3. One anchor: falls through to single-sided handling.
        An anchor exactly at center counts as the right side.

    Outputs:
        (left, right) ordered by x; either may be None.
    """
    left_side = sorted((a for a in anchors if a.foot_x < center_x),
                       key=lambda a: -a.foot_x)   # nearest center first
    right_side = sorted((a for a in anchors if a.foot_x >= center_x),
                        key=lambda a: a.foot_x)   # nearest center first

    if left_side and right_side:
        return left_side[0], right_side[0]
    if len(right_side) >= 2:
        return right_side[0], right_side[1]
    if len(left_side) >= 2:
        return left_side[1], left_side[0]
    if right_side:
        return None, right_side[0]
    if left_side:
        return left_side[0], None
    return None, None

def _single_sided(
        anchor: BoundaryAnchor,
        side: str,
        center_x: float,
        config: LaneOffsetConfig,
        frame_id: int,
        timestamp_ms: int,
        boundary_count: int,
        log: list[str],
    ) -> LaneOffsetResult:
    """
    Offset from one boundary, by projecting where the lane center must be.

    Purpose:
        Returns the same quantity as two-boundary mode (deviation from the
        lane center), not distance from the visible line. Without
        expected_half_lane_px those would be different quantities on
        different scales feeding the same controller, so the mode reports
        "single_uncalibrated" with zero confidence until it's set.

    Side effects:
        Appends to log when uncalibrated.
    """
    half = config.expected_half_lane_px
    if half is None:
        log.append(
            "[UNCALIBRATED] one usable boundary, but expected_half_lane_px is "
            "unset — emitting no steering signal rather than one of unknown scale"
        )
        return LaneOffsetResult(
            offset = 0.0, left_x = None, right_x = None, lane_width_px = None,
            confidence = 0.0, boundary_count = boundary_count,
            mode = MODE_SINGLE_UNCALIBRATED,
            frame_id = frame_id, timestamp_ms = timestamp_ms,
        )

    if side == "left":
        implied_center = anchor.foot_x + half
        mode, left_x, right_x = MODE_LEFT_ONLY, anchor.foot_x, None
    else:
        implied_center = anchor.foot_x - half
        mode, left_x, right_x = MODE_RIGHT_ONLY, None, anchor.foot_x

    offset = clamp((center_x - implied_center) / center_x, -1.0, 1.0)
    return LaneOffsetResult(
        offset = round(offset, 4),
        left_x = left_x,
        right_x = right_x,
        lane_width_px = None,
        confidence = round(anchor.weight, 4),
        boundary_count = boundary_count,
        mode = mode,
        frame_id = frame_id,
        timestamp_ms = timestamp_ms,
    )


def compute_lane_offset(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
        config: LaneOffsetConfig = LaneOffsetConfig(),
    ) -> tuple[LaneOffsetResult, dict]:
    """
    Estimate the robot's lateral offset from its own lane center.

    Inputs:
        geometry: From run_geometry_stage().
        roi: Supplies the lane ROI width (the robot sits at half of it) and
            the frame stamp.

    Outputs:
        (result, debug_summary). Anchors are in lane-ROI px; add
        roi.lane_rect[0] for frame coordinates. debug_summary holds frame_id,
        timestamp_ms, mode, raw_count, usable_count, anchors
        ([(foot_x, weight)]) and log.

    Raises:
        ValueError: If either input is None, or their frame stamps disagree.
    """
    check_same_frame(geometry, roi, "compute_lane_offset")

    log = []
    frame_id = geometry.frame_id
    timestamp_ms = geometry.timestamp_ms

    roi_width = roi.lane_rect[2]
    center_x = roi_width / 2.0

    raw = list(geometry.lane_candidates)
    usable = [c for c in raw if _usable(c, config, log)]
    anchors = [_anchor(c, config) for c in usable]
    boundary_count = len(anchors)

    def _summary(result):
        return result, {
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "mode": result.mode,
            "raw_count": len(raw),
            "usable_count": boundary_count,
            "anchors": [(round(a.foot_x, 1), a.weight) for a in anchors],
            "log": log,
        }

    if not anchors:
        if raw:
            log.append(f"[BLIND] {len(raw)} candidates, none usable as a boundary")
        return _summary(LaneOffsetResult(
            offset = 0.0, left_x = None, right_x = None, lane_width_px = None,
            confidence = 0.0, boundary_count = 0, mode = MODE_NONE,
            frame_id = frame_id, timestamp_ms = timestamp_ms,
        ))

    left, right = _lane_pair(anchors, center_x)

    if left is not None and right is not None:
        # Check the spacing before trusting the pair. Implausible spacing falls
        # back to the strongest single anchor.
        lane_width_px = right.foot_x - left.foot_x

        if lane_width_px < config.min_lane_width_px:
            log.append(
                f"[MERGE] anchors {lane_width_px:.1f}px apart, below "
                f"{config.min_lane_width_px} — same marking, not opposite boundaries"
            )
            best = max(anchors, key=lambda a: a.weight)
            side = "left" if best.foot_x < center_x else "right"
            return _summary(_single_sided(best, side, center_x, config,
                                          frame_id, timestamp_ms,
                                          boundary_count, log))

        if lane_width_px > config.max_lane_width_px:
            log.append(
                f"[SPAN] anchors {lane_width_px:.1f}px apart, above "
                f"{config.max_lane_width_px} — pair spans more than one lane"
            )
            best = max(anchors, key=lambda a: a.weight)
            side = "left" if best.foot_x < center_x else "right"
            return _summary(_single_sided(best, side, center_x, config,
                                          frame_id, timestamp_ms,
                                          boundary_count, log))

        lane_center = (left.foot_x + right.foot_x) / 2.0
        offset = clamp((center_x - lane_center) / center_x, -1.0, 1.0)
        total_weight = left.weight + right.weight
        mean_weight = total_weight / 2.0 if total_weight > 0 else 0.0

        return _summary(LaneOffsetResult(
            offset = round(offset, 4),
            left_x = round(left.foot_x, 2),
            right_x = round(right.foot_x, 2),
            lane_width_px = round(lane_width_px, 2),
            confidence = round(mean_weight, 4),
            boundary_count = boundary_count,
            mode = MODE_TWO_BOUNDARY,
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
        ))

    anchor = left if left is not None else right
    side = "left" if left is not None else "right"
    log.append(f"[ONE-SIDED] only a {side} boundary is usable this frame")
    return _summary(_single_sided(anchor, side, center_x, config,
                                  frame_id, timestamp_ms, boundary_count, log))