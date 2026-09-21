"""
lane_offset.py

Lane Offset Estimation

Purpose:
    Compute the lateral offset of the robot from the center of its own lane,
    using the lane boundary candidates produced by the geometry branch.

    This stage consumes GeometryBranchResult rather than fusion output.
    Fusion normalizes detections into the Phase 2 publishing contract and
    keeps only a bbox centroid and a confidence. Estimation needs the geometry
    that normalization discards. The two are siblings over the same frame, not
    a sequence:

        crop_rois -> geometry -+-> fuse_detections     -> Phase2Output
                               +-> compute_lane_offset -> steering error

Offsets:
     0.0: robot is centered in its lane
    -1.0: robot is fully left of the lane center
    +1.0: robot is fully right of the lane center

    Normalized to the lane ROI width, so it stays resolution-independent and
    is usable directly as a Phase 3 steering error.

What each candidate field is used for:
    contour        the anchor x is the mean x of the contour points in its
                   lowest rows --- where the marking sits at its nearest point
                   to the robot. A bbox centroid puts an angled line's anchor
                   halfway up the ROI, which is not where the robot is about
                   to arrive
    proximity      gates out candidates too far up the ROI to be near-field
                   evidence, and weights the ones that remain, so a boundary
                   at the bottom of the ROI counts for more than one at the top
    width_px       minAreaRect short side. A marking far wider than a painted
                   line is a blob, a glare patch or a merged pair, not a
                   boundary to steer by
    length_px      minAreaRect long side. Short fragments cannot anchor a
                   boundary; this is the gate that keeps dashed-center-line
                   fragments from being treated as lane edges
    mean_intensity separates real bright tape from dim contours picked up off
                   shadow edges and mat seams

Boundary selection:
    The robot sits at the horizontal center of the lane ROI, so its own lane
    is bounded by the nearest usable boundary on each side of ROI center, not
    by the outermost pair. On a two-lane street with a dividing line visible,
    taking the extremes centers the robot on the street rather than on its
    lane, which is a systematically directional error.

Calibration seam:
    Single-boundary mode needs to know how far the lane center sits from a
    boundary. That number comes from camera calibration (cm-per-pixel against
    a known lane width), so LaneOffsetConfig.expected_half_lane_px has no
    default. Until it is set, a frame with only one usable boundary reports
    mode "single_uncalibrated" with zero confidence rather than emitting a
    steering error whose scale is unknown.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from src.perception.geometry import GeometryBranchResult, LaneCandidate
from src.perception.roi_crop import ROICropResult

# =============================================================================
# Input Dataclass
# =============================================================================
@dataclass(frozen=True)
class LaneOffsetConfig:
    """
    Tuning for lane offset estimation

    These gates are a second, stricter pass than LaneContourFilter. Geometry
    decides "is this a lane marking"; this decides "is this trustworthy enough
    to steer by". A candidate can legitimately pass the first and fail this.

    conf_threshold: minimum candidate confidence to anchor a boundary
    min_proximity: minimum proximity [0,1]. Below this the candidate sits too
                   far up the ROI to describe where the robot is now
    min_length_px: minimum minAreaRect long side for a usable boundary
    min_width_px: minimum minAreaRect short side; below this is noise
    max_width_px: maximum short side; above this the contour is a blob or a
                  merged pair rather than a single marking
    min_intensity: minimum mean intensity inside the bbox
    min_lane_width_px: minimum spacing between two boundaries for them to be
                       opposite sides of one lane. Two contours closer than
                       this are fragments of the same marking
    max_lane_width_px: maximum spacing. Wider than this and the pair spans
                       more than one lane
    expected_half_lane_px: distance from a boundary to the lane center, in
                           lane-ROI pixels. From calibration; see the module
                           docstring. None disables single-boundary mode
    foot_band_px: height of the band at the bottom of a contour used to
                  compute its anchor x
    """
    conf_threshold: float = 0.30
    min_proximity: float = 0.25
    min_length_px: float = 25.0
    min_width_px: float = 1.0
    max_width_px: float = 25.0
    min_intensity: float = 90.0
    min_lane_width_px: float = 60.0
    max_lane_width_px: float = 400.0
    expected_half_lane_px: Optional[float] = None
    foot_band_px: int = 6

# =============================================================================
# Output Dataclasses
# =============================================================================
@dataclass(frozen=True)
class BoundaryAnchor:
    """
    One lane boundary reduced to the quantities the offset math uses

    foot_x: anchor x in lane-ROI pixels, from the contour's lowest rows
    weight: confidence scaled by proximity; how much this anchor counts
    candidate: the LaneCandidate it came from, kept for debug and for the
               scene state machine to reach back into later
    """
    foot_x: float
    weight: float
    candidate: LaneCandidate

@dataclass(frozen=True)
class LaneOffsetResult:
    """
    Lane offset estimate for a single frame

    offset: normalized lateral offset from the lane center
            Negative = robot is left of lane center
            Positive = robot is right of lane center
    left_x: anchor x of the left boundary, None if unused
    right_x: anchor x of the right boundary, None if unused
    lane_width_px: spacing between the two anchors, None in single-sided modes
    confidence: weighted confidence of the anchors actually used
    boundary_count: lane_boundary candidates that passed every usability gate
    mode: "two_boundary" | "left_only" | "right_only" |
          "single_uncalibrated" | "none"
    frame_id: carried from GeometryBranchResult, never re-derived
    timestamp_ms: carried from GeometryBranchResult, never re-derived

    boundary_count reports usable boundaries, not raw detections. A frame with
    four contours and no usable boundary reports 0 here, which is what
    distinguishes "saw nothing" from "saw nothing it could steer by".
    """
    offset: float
    left_x: Optional[float]
    right_x: Optional[float]
    lane_width_px: Optional[float]
    confidence: float
    boundary_count: int
    mode: str
    frame_id: int
    timestamp_ms: int

# =============================================================================
# Utility Functions
# =============================================================================
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def foot_x(
        candidate: LaneCandidate,
        band_px: int = 6
    ) -> float:
    """
    Purpose:
        Anchor x for a boundary: the mean x of the contour points sitting in
        the lowest band of that contour. This is where the marking is at its
        closest approach to the robot

    Inputs:
        candidate: LaneCandidate with an ROI-local contour
        band_px: height of the band measured up from the contour's lowest point

    Outputs:
        anchor x in lane-ROI pixels

    Notes:
        Falls back to the bbox horizontal center when no contour is present,
        so hand-built candidates still work. A near-vertical marking gives the
        same answer either way; an angled one does not, which is the whole
        reason this exists
    """
    # Prefer the anchor geometry computed from the full contour. It owns the
    # contour, so it can measure this without the candidate having to carry
    # one -- which matters for CSV replay, where the contour is gone.
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
        log: list,
    ) -> bool:
    """
    Purpose:
        Decide whether a candidate is trustworthy enough to anchor a boundary,
        logging the specific gate that rejected it

    Notes:
        Every rejection is logged with its gate name so the debug summary
        answers "why did this frame go blind" without a rerun
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
    """
    Purpose:
        Reduce a usable candidate to its anchor x and steering weight

    Notes:
        weight blends confidence with proximity rather than replacing it. A
        high-confidence marking at the top of the ROI is real, it just says
        less about where the robot is right now
    """
    weight = candidate.confidence * (0.5 + 0.5 * candidate.proximity)
    return BoundaryAnchor(
        foot_x = foot_x(candidate, config.foot_band_px),
        weight = round(weight, 4),
        candidate = candidate,
    )

def _lane_pair(
        anchors: List[BoundaryAnchor],
        center_x: float,
    ) -> tuple:
    """
    Purpose:
        Select the two boundaries of the lane nearest the robot

    Outputs:
        (left, right) ordered by x, either of which may be None

    Rules:
        1. If anchors exist on both sides of ROI center, take the nearest on
           each side. Those bracket the robot, so they are its own lane
        2. If every anchor is on one side, the robot has drifted out of the
           lane. The two nearest center still describe the nearest lane, and
           the resulting offset is the error that steers back. Requiring a
           straddling pair would discard a usable measurement on exactly the
           frames where it matters most
        3. One anchor total falls through to single-sided handling

    Notes:
        Rule 1 is what separates the robot's lane from the street. Taking the
        outermost pair instead centers the robot on whatever the widest pair
        of markings spans, which on a two-lane street with a visible dividing
        line is the street center --- an error that always points the same
        direction
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
        log: list,
    ) -> LaneOffsetResult:
    """
    Purpose:
        Offset from one boundary, by projecting where the lane center must be

    Notes:
        Returns the same quantity as two-boundary mode --- deviation from the
        lane center --- rather than distance from the visible line. Without
        expected_half_lane_px those are different quantities on different
        scales feeding the same PID, so the mode is disabled until it is set
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
            mode = "single_uncalibrated",
            frame_id = frame_id, timestamp_ms = timestamp_ms,
        )

    if side == "left":
        implied_center = anchor.foot_x + half
        mode, left_x, right_x = "left_only", anchor.foot_x, None
    else:
        implied_center = anchor.foot_x - half
        mode, left_x, right_x = "right_only", None, anchor.foot_x

    offset = _clamp((center_x - implied_center) / center_x, -1.0, 1.0)
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

# =============================================================================
# Lane Offset Stage
# =============================================================================
def compute_lane_offset(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
        config: LaneOffsetConfig = LaneOffsetConfig(),
    ) -> tuple:
    """
    Purpose:
        Estimate the robot's lateral offset from its own lane center

    Inputs:
        geometry: GeometryBranchResult from run_geometry_stage()
        roi: ROICropResult, for the lane ROI width and the frame stamp
        config: LaneOffsetConfig tuning

    Outputs:
        result: LaneOffsetResult
        debug_summary: dict --- "frame_id", "timestamp_ms", "mode",
                       "raw_count", "usable_count", "anchors", "log"

    Notes:
        Anchors and offsets are in lane-ROI pixel coordinates. Add
        roi.lane_rect[0] to convert an anchor x to frame coordinates
    """
    if geometry is None:
        raise ValueError("compute_lane_offset: geometry result is None")
    if roi is None:
        raise ValueError("compute_lane_offset: roi result is None")
    if (geometry.frame_id, geometry.timestamp_ms) != (roi.frame_id, roi.timestamp_ms):
        raise ValueError(
            f"compute_lane_offset: geometry stamp "
            f"{(geometry.frame_id, geometry.timestamp_ms)} does not match roi stamp "
            f"{(roi.frame_id, roi.timestamp_ms)} — candidates are from different frames"
        )

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

    # ========================================================================
    # No usable boundary
    # ========================================================================
    if not anchors:
        if raw:
            log.append(f"[BLIND] {len(raw)} candidates, none usable as a boundary")
        return _summary(LaneOffsetResult(
            offset = 0.0, left_x = None, right_x = None, lane_width_px = None,
            confidence = 0.0, boundary_count = 0, mode = "none",
            frame_id = frame_id, timestamp_ms = timestamp_ms,
        ))

    # ========================================================================
    # Select the boundaries of the robot's own lane
    # ========================================================================
    left, right = _lane_pair(anchors, center_x)

    # ========================================================================
    # Two boundaries - validate the spacing before trusting the pair
    # ========================================================================
    if left is not None and right is not None:
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
        offset = _clamp((center_x - lane_center) / center_x, -1.0, 1.0)
        total_weight = left.weight + right.weight
        mean_weight = total_weight / 2.0 if total_weight > 0 else 0.0

        return _summary(LaneOffsetResult(
            offset = round(offset, 4),
            left_x = round(left.foot_x, 2),
            right_x = round(right.foot_x, 2),
            lane_width_px = round(lane_width_px, 2),
            confidence = round(mean_weight, 4),
            boundary_count = boundary_count,
            mode = "two_boundary",
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
        ))

    # ========================================================================
    # One side only
    # ========================================================================
    anchor = left if left is not None else right
    side = "left" if left is not None else "right"
    log.append(f"[ONE-SIDED] only a {side} boundary is usable this frame")
    return _summary(_single_sided(anchor, side, center_x, config,
                                  frame_id, timestamp_ms, boundary_count, log))