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

from geometry import GeometryBranchResult, LaneCandidate
from roi_crop import ROICropResult

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

# =============================================================================
# Self-test
#
#   python3 lane_offset.py
#
# Exits non-zero on any logic failure so it can gate a commit.
# =============================================================================
if __name__ == "__main__":
    import sys
    import traceback

    from capture import FrameData
    from preprocess import preprocess_frame
    from roi_crop import crop_rois

    FID, TS = 9, 2000

    _results = []

    def check(name, fn):
        """Run one test, record pass/fail, never abort the suite."""
        try:
            fn()
        except Exception:
            _results.append((name, False))
            print(f"  FAIL  {name}")
            for line in traceback.format_exc().strip().splitlines()[-2:]:
                print(f"        {line.strip()}")
        else:
            _results.append((name, True))
            print(f"  ok    {name}")

    def expect_raises(exc_type, fn):
        try:
            fn()
        except exc_type:
            return
        raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")

    def _roi(frame_id=FID, timestamp_ms=TS):
        """An ROICropResult built by running the real stages in order."""
        rng = np.random.default_rng(6)
        bgr = rng.integers(0, 256, (360, 480, 3), dtype=np.uint8)
        return crop_rois(preprocess_frame(FrameData(bgr, frame_id, timestamp_ms)))

    ROI_W = 432          # lane_rect width at 480x360
    CENTER = ROI_W / 2.0 # 216.0

    def _mark(x, y=95, length=90.0, width=4.0, conf=0.8, prox=0.9,
              intensity=200.0, skew=0):
        """
        A lane marking whose contour runs from (x, y-length) up to (x+skew, y).
        skew lets a test separate the bbox centroid from the contour foot.
        """
        pts = []
        for i in range(0, int(length) + 1, 3):
            t = i / max(length, 1)
            pts.append([[int(x + skew * t), int(y - length + i)]])
        contour = np.array(pts, dtype=np.int32)
        xs = contour[:, 0, 0]
        ys = contour[:, 0, 1]
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)
        return LaneCandidate(
            label="lane_boundary", bbox=bbox, contour=contour, confidence=conf,
            frame_id=FID, timestamp_ms=TS, proximity=prox,
            width_px=width, length_px=length, mean_intensity=intensity,
        )

    def _geo(*lanes, frame_id=FID, timestamp_ms=TS):
        return GeometryBranchResult(list(lanes), [], frame_id, timestamp_ms)

    def _run(*lanes, config=None):
        return compute_lane_offset(_geo(*lanes), _roi(),
                                   config or LaneOffsetConfig())

    # -------------------------------------------------------------------------
    # Anchoring from the contour
    # -------------------------------------------------------------------------
    def t_foot_x_uses_the_bottom_of_the_contour():
        """An angled marking anchors where it is nearest the robot."""
        angled = _mark(150, skew=40)
        bbox_center = angled.bbox[0] + angled.bbox[2] / 2.0
        anchored = foot_x(angled)
        assert anchored > bbox_center + 10, (
            f"foot {anchored:.1f} should sit right of bbox center {bbox_center:.1f}"
        )

    def t_foot_x_falls_back_to_bbox():
        c = _mark(150)
        no_contour = LaneCandidate(
            label="lane_boundary", bbox=c.bbox, contour=None, confidence=0.8,
            frame_id=FID, timestamp_ms=TS, proximity=0.9,
            width_px=4.0, length_px=90.0, mean_intensity=200.0,
        )
        assert foot_x(no_contour) == c.bbox[0] + c.bbox[2] / 2.0

    # -------------------------------------------------------------------------
    # Two-boundary offset
    # -------------------------------------------------------------------------
    def t_centered_robot_reads_zero():
        r, _ = _run(_mark(CENTER - 70), _mark(CENTER + 70))
        assert abs(r.offset) < 0.01, f"got {r.offset}"
        assert r.mode == "two_boundary"

    def t_lane_width_is_positive():
        """left_x - right_x was negative for every frame in the old version."""
        r, _ = _run(_mark(CENTER - 70), _mark(CENTER + 70))
        assert r.lane_width_px > 0, f"got {r.lane_width_px}"
        assert abs(r.lane_width_px - 140) < 2, f"got {r.lane_width_px}"

    # -------------------------------------------------------------------------
    # Boundary selection
    # -------------------------------------------------------------------------
    def t_three_boundaries_pick_the_robots_own_lane():
        """Outer-left edge, dividing line, outer-right edge. The robot is in
        the right lane, so its boundaries are the dividing line and the outer
        right edge --- not the two outermost markings."""
        r, _ = _run(_mark(40), _mark(200), _mark(390))
        assert r.mode == "two_boundary"
        assert (r.left_x, r.right_x) == (200.0, 390.0), (
            f"picked {(r.left_x, r.right_x)}; extremes would be (40, 390)"
        )
        assert r.offset < 0, "robot is left of the right lane's center"

    def t_drifted_out_of_lane_still_measures():
        """Both boundaries right of the camera: the robot is left of the lane
        and the pair still gives the error that steers back."""
        r, _ = _run(_mark(CENTER + 10), _mark(CENTER + 150))
        assert r.mode == "two_boundary", f"got {r.mode}"
        assert r.offset < -0.1, f"got {r.offset}"

    def t_drifted_the_other_way_still_measures():
        r, _ = _run(_mark(CENTER - 150), _mark(CENTER - 10))
        assert r.mode == "two_boundary", f"got {r.mode}"
        assert r.offset > 0.1, f"got {r.offset}"

    def t_extremes_would_have_read_centered():
        """The old behaviour on the same frame, as a contrast."""
        r, _ = _run(_mark(40), _mark(200), _mark(390))
        extremes_center = (40 + 390) / 2.0
        assert abs(extremes_center - CENTER) < 3, "fixture no longer demonstrates it"
        assert abs(r.offset) > 0.1, (
            "the outermost pair reads centered; the bracketing pair must not"
        )

    # -------------------------------------------------------------------------
    # Width validation
    # -------------------------------------------------------------------------
    def t_fragments_are_not_opposite_boundaries():
        """Two pieces of one broken line 6px apart used to become a lane."""
        r, dbg = _run(_mark(CENTER - 3), _mark(CENTER + 3))
        assert r.mode != "two_boundary", f"got {r.mode}"
        assert any("[MERGE]" in e for e in dbg["log"]), dbg["log"]

    def t_pair_spanning_two_lanes_rejected():
        cfg = LaneOffsetConfig(max_lane_width_px=200.0)
        r, dbg = _run(_mark(30), _mark(400), config=cfg)
        assert r.mode != "two_boundary"
        assert any("[SPAN]" in e for e in dbg["log"]), dbg["log"]

    # -------------------------------------------------------------------------
    # Geometry gates
    # -------------------------------------------------------------------------
    def t_short_fragment_rejected():
        r, dbg = _run(_mark(CENTER - 70), _mark(CENTER + 70, length=9.0))
        assert r.boundary_count == 1, f"got {r.boundary_count}"
        assert any("length_px" in e for e in dbg["log"])

    def t_wide_blob_rejected():
        r, dbg = _run(_mark(CENTER - 70), _mark(CENTER + 70, width=60.0))
        assert r.boundary_count == 1
        assert any("width_px" in e for e in dbg["log"])

    def t_dim_contour_rejected():
        r, dbg = _run(_mark(CENTER - 70), _mark(CENTER + 70, intensity=40.0))
        assert r.boundary_count == 1
        assert any("mean_intensity" in e for e in dbg["log"])

    def t_far_up_the_roi_rejected():
        r, dbg = _run(_mark(CENTER - 70), _mark(CENTER + 70, prox=0.05))
        assert r.boundary_count == 1
        assert any("proximity" in e for e in dbg["log"])

    def t_low_confidence_rejected():
        r, dbg = _run(_mark(CENTER - 70), _mark(CENTER + 70, conf=0.1))
        assert r.boundary_count == 1
        assert any("confidence" in e for e in dbg["log"])

    def t_proximity_weights_the_confidence():
        near, _ = _run(_mark(CENTER - 70, prox=1.0), _mark(CENTER + 70, prox=1.0))
        far, _ = _run(_mark(CENTER - 70, prox=0.3), _mark(CENTER + 70, prox=0.3))
        assert near.confidence > far.confidence, (
            f"near {near.confidence} should outweigh far {far.confidence}"
        )

    # -------------------------------------------------------------------------
    # Single-sided behaviour
    # -------------------------------------------------------------------------
    def t_one_boundary_uncalibrated_emits_nothing():
        r, dbg = _run(_mark(CENTER - 70))
        assert r.mode == "single_uncalibrated", f"got {r.mode}"
        assert r.offset == 0.0 and r.confidence == 0.0
        assert any("[UNCALIBRATED]" in e for e in dbg["log"])

    def t_one_boundary_calibrated_projects_the_lane_center():
        """With the half-lane known, a left line at center-70 and a half-lane
        of 70 means the robot is centered."""
        cfg = LaneOffsetConfig(expected_half_lane_px=70.0)
        r, _ = _run(_mark(CENTER - 70), config=cfg)
        assert r.mode == "left_only", f"got {r.mode}"
        assert abs(r.offset) < 0.01, f"got {r.offset}"

    def t_single_and_two_boundary_agree():
        """Both modes must return the same quantity or the PID sees a step."""
        cfg = LaneOffsetConfig(expected_half_lane_px=70.0)
        two, _ = _run(_mark(CENTER - 40), _mark(CENTER + 100), config=cfg)
        one, _ = _run(_mark(CENTER - 40), config=cfg)
        assert abs(two.offset - one.offset) < 0.02, (
            f"two_boundary {two.offset} vs left_only {one.offset}"
        )

    def t_blind_frame_reports_none():
        r, _ = _run()
        assert r.mode == "none" and r.offset == 0.0 and r.boundary_count == 0

    def t_all_candidates_rejected_is_not_the_same_as_blind():
        r, dbg = _run(_mark(CENTER, conf=0.05))
        assert r.mode == "none" and r.boundary_count == 0
        assert dbg["raw_count"] == 1, "the raw detection must still be reported"
        assert any("[BLIND]" in e for e in dbg["log"])

    # -------------------------------------------------------------------------
    # Stamp and contract
    # -------------------------------------------------------------------------
    def t_stamp_is_carried_not_rederived():
        r, dbg = _run(_mark(CENTER - 70), _mark(CENTER + 70))
        assert (r.frame_id, r.timestamp_ms) == (FID, TS)
        assert (dbg["frame_id"], dbg["timestamp_ms"]) == (FID, TS)

    def t_cross_frame_inputs_rejected():
        expect_raises(ValueError, lambda: compute_lane_offset(
            _geo(frame_id=1, timestamp_ms=100), _roi(frame_id=2, timestamp_ms=200)
        ))

    def t_result_is_immutable():
        r, _ = _run(_mark(CENTER - 70), _mark(CENTER + 70))
        expect_raises(Exception, lambda: setattr(r, "offset", 0.5))

    def t_none_inputs_rejected():
        expect_raises(ValueError, lambda: compute_lane_offset(None, _roi()))
        expect_raises(ValueError, lambda: compute_lane_offset(_geo(), None))

    # -------------------------------------------------------------------------
    print("\nAnchoring")
    check("foot_x uses the contour bottom",      t_foot_x_uses_the_bottom_of_the_contour)
    check("foot_x falls back to bbox",           t_foot_x_falls_back_to_bbox)

    print("\nTwo-boundary offset")
    check("centered robot reads zero",           t_centered_robot_reads_zero)
    check("lane_width_px is positive",           t_lane_width_is_positive)
    check("lane right of camera -> negative",    t_drifted_out_of_lane_still_measures)
    check("lane left of camera -> positive",     t_drifted_the_other_way_still_measures)

    print("\nBoundary selection")
    check("three boundaries pick own lane",      t_three_boundaries_pick_the_robots_own_lane)
    check("extremes would have read centered",   t_extremes_would_have_read_centered)

    print("\nWidth validation")
    check("fragments are not a lane",            t_fragments_are_not_opposite_boundaries)
    check("pair spanning two lanes rejected",    t_pair_spanning_two_lanes_rejected)

    print("\nGeometry gates")
    check("short fragment rejected",             t_short_fragment_rejected)
    check("wide blob rejected",                  t_wide_blob_rejected)
    check("dim contour rejected",                t_dim_contour_rejected)
    check("far up the ROI rejected",             t_far_up_the_roi_rejected)
    check("low confidence rejected",             t_low_confidence_rejected)
    check("proximity weights confidence",        t_proximity_weights_the_confidence)

    print("\nSingle-sided")
    check("uncalibrated emits no signal",        t_one_boundary_uncalibrated_emits_nothing)
    check("calibrated projects lane center",     t_one_boundary_calibrated_projects_the_lane_center)
    check("single and two-boundary agree",       t_single_and_two_boundary_agree)
    check("blind frame reports none",            t_blind_frame_reports_none)
    check("all-rejected is not blind",           t_all_candidates_rejected_is_not_the_same_as_blind)

    print("\nStamp and contract")
    check("frame_id/timestamp carried through",  t_stamp_is_carried_not_rederived)
    check("cross-frame inputs rejected",         t_cross_frame_inputs_rejected)
    check("LaneOffsetResult is immutable",       t_result_is_immutable)
    check("None inputs rejected",                t_none_inputs_rejected)

    passed = sum(1 for _, ok in _results if ok)
    print(f"\n{passed}/{len(_results)} passed")

    # -------------------------------------------------------------------------
    # CSV sweep: replay the geometry branch's dataset output
    #
    #   python3 lane_offset.py --csv
    #   python3 lane_offset.py --csv path/to/lane_candidates.csv
    #   python3 lane_offset.py --csv [path] --video [--frames DIR]
    #
    # Answers the questions the unit tests cannot: on real candidates, which
    # gate is doing the rejecting, how often the stage goes blind, and what
    # the gated fields actually range over. Set the gates from these numbers
    # rather than from the course dimensions.
    #
    # --video renders every replayed frame with its decision drawn on it
    # (debug_video.py) to results/<csv name>_lane_offset.avi, plus a per-frame
    # CSV sidecar. --frames DIR supplies the source images, matched to
    # frame_id by the last number in each filename; frames with no match are
    # drawn on a blank canvas so the decision is still reviewable.
    #
    # Limitation: the CSV has no contour column, so every anchor falls back to
    # the bbox horizontal center. If the CSV carries an optional foot_x column
    # it is used instead, via a one-point contour. Otherwise the contour path
    # is exercised by the unit tests above, not here.
    # -------------------------------------------------------------------------
    def _flag_value(flag, default=None):
        """Value following flag, or default if the flag has no value."""
        i = sys.argv.index(flag)
        if len(sys.argv) > i + 1 and not sys.argv[i + 1].startswith("--"):
            return sys.argv[i + 1]
        return default

    if "--csv" not in sys.argv:
        note = "; --video needs --csv" if "--video" in sys.argv else ""
        print(f"\nCSV sweep: skipped (pass --csv to replay the dataset{note})")
        sys.exit(0 if passed == len(_results) else 1)

    import csv as _csv
    import os

    csv_path = _flag_value("--csv", "vision_stack/frames/lane_candidates.csv")

    if not os.path.isfile(csv_path):
        print(f"\nCSV sweep: {csv_path} not found")
        sys.exit(0 if passed == len(_results) else 1)

    NEEDED = ("frame_id", "confidence", "mean_intensity", "length_px",
              "width_px", "proximity", "x", "y", "w", "h")

    rows = []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        missing = [c for c in NEEDED if c not in (reader.fieldnames or [])]
        if missing:
            print(f"\nCSV sweep: {csv_path} is missing columns {missing}")
            sys.exit(1)
        for row in reader:
            rows.append(row)

    # Group rows into frames, preserving file order
    frames = {}
    order = []
    for row in rows:
        fid = int(row["frame_id"])
        if fid not in frames:
            frames[fid] = []
            order.append(fid)
        frames[fid].append(row)

    def _candidate(row):
        """
        A LaneCandidate from one CSV row. With no foot_x column the contour
        is absent and foot_x() falls back to the bbox center. With one, a
        single point at (foot_x, bbox bottom) makes foot_x() return it exactly.
        """
        bbox = (int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
        contour = None
        if row.get("foot_x") not in (None, ""):
            contour = np.array([[[float(row["foot_x"]), bbox[1] + bbox[3] - 1]]])
        return LaneCandidate(
            label = "lane_boundary",
            bbox = bbox,
            contour = contour,
            confidence = float(row["confidence"]),
            frame_id = int(row["frame_id"]),
            timestamp_ms = 0,
            proximity = float(row["proximity"]),
            width_px = float(row["width_px"]),
            length_px = float(row["length_px"]),
            mean_intensity = float(row["mean_intensity"]),
        )

    def _fingerprint(frame_rows):
        """Candidate geometry only, so a stuck capture looks identical here."""
        return tuple(sorted(
            (r["confidence"], r["mean_intensity"], r["length_px"],
             r["width_px"], r["proximity"], r["x"], r["y"], r["w"], r["h"])
            for r in frame_rows
        ))

    config = LaneOffsetConfig()
    lane_rect = (24, 252, 432, 108)   # 480x360; adjust if the sweep is from
                                      # a different capture resolution

    blank_roi = ROICropResult(
        lane_roi = np.zeros((lane_rect[3], lane_rect[2]), np.uint8),
        traffic_roi = np.zeros((180, 240, 3), np.uint8),
        sign_roi = np.zeros((198, 240), np.uint8),
        lane_rect = lane_rect,
        traffic_rect = (120, 0, 240, 180),
        sign_rect = (240, 0, 240, 198),
        frame_id = 0, timestamp_ms = 0, source_shape = (360, 480),
    )

    # -------------------------------------------------------------------------
    # Debug video (optional). Imported here so the unit tests and the plain
    # sweep keep running without it
    # -------------------------------------------------------------------------
    video = None
    frame_index = {}
    video_stats = {"source": 0, "blank": 0, "unreadable": 0}
    VIDEO_SCALE = 2     # 480x360 -> 960x720; gate labels are unreadable at 1x

    if "--video" in sys.argv:
        import re
        from debug_video import (DebugVideoWriter, IMAGE_EXTENSIONS,
                                 load_source_frame)

        frames_dir = _flag_value("--frames") if "--frames" in sys.argv else None
        if frames_dir is not None:
            if not os.path.isdir(frames_dir):
                print(f"\nCSV sweep: --frames {frames_dir} is not a directory")
                sys.exit(1)
            dup_ids = 0
            for name in sorted(os.listdir(frames_dir)):
                fstem, ext = os.path.splitext(name)
                nums = re.findall(r"\d+", fstem)
                if ext.lower() not in IMAGE_EXTENSIONS or not nums:
                    continue
                fid_key = int(nums[-1])
                if fid_key in frame_index:
                    dup_ids += 1        # first file wins
                    continue
                frame_index[fid_key] = os.path.join(frames_dir, name)
            if dup_ids:
                print(f"\n[VIDEO] {dup_ids} images share a frame number with "
                      "an earlier file; the first one is used")

        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results")
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        video = DebugVideoWriter(
            os.path.join(results_dir, f"{stem}_lane_offset.avi"))

    modes = {}
    gate_hits = {}
    offsets = []
    widths = []
    usable_counts = {}
    blind_run = cur_run = 0
    blind_start = cur_start = None
    dup_runs = []
    prev_fp = None
    cur_dup = 1

    field_values = {"confidence": [], "proximity": [], "length_px": [],
                    "width_px": [], "mean_intensity": []}

    for i, fid in enumerate(order):
        frame_rows = frames[fid]
        cands = [_candidate(r) for r in frame_rows]
        for c in cands:
            field_values["confidence"].append(c.confidence)
            field_values["proximity"].append(c.proximity)
            field_values["length_px"].append(c.length_px)
            field_values["width_px"].append(c.width_px)
            field_values["mean_intensity"].append(c.mean_intensity)

        roi = ROICropResult(
            **{**blank_roi.__dict__, "frame_id": fid, "timestamp_ms": 0}
        )
        result, dbg = compute_lane_offset(
            GeometryBranchResult(cands, [], fid, 0), roi, config
        )

        if video is not None:
            # Same gate function the stage used, one candidate at a time, so
            # each box is labeled with the gate that actually rejected it
            marks = []
            for c in cands:
                scratch = []
                gate = None if _usable(c, config, scratch) else scratch[0].split()[1]
                marks.append((c.bbox, gate))

            src = None
            if fid in frame_index:
                src = load_source_frame(frame_index[fid], lane_rect)
                video_stats["unreadable" if src is None else "source"] += 1
            if src is None:
                video_stats["blank"] += 1
            try:
                video.write(src, result, dbg, lane_rect, marks, scale=VIDEO_SCALE)
            except RuntimeError as exc:
                print(f"\n[VIDEO] {exc}")
                sys.exit(1)

        modes[result.mode] = modes.get(result.mode, 0) + 1
        usable_counts[result.boundary_count] = \
            usable_counts.get(result.boundary_count, 0) + 1

        for entry in dbg["log"]:
            if entry.startswith("[REJECT]"):
                gate = entry.split()[1]
                gate_hits[gate] = gate_hits.get(gate, 0) + 1

        if result.mode in ("two_boundary", "left_only", "right_only"):
            offsets.append(result.offset)
            if result.lane_width_px is not None:
                widths.append(result.lane_width_px)
            cur_run = 0
            cur_start = None
        else:
            if cur_run == 0:
                cur_start = fid
            cur_run += 1
            if cur_run > blind_run:
                blind_run, blind_start = cur_run, cur_start

        fp = _fingerprint(frame_rows)
        if fp == prev_fp:
            cur_dup += 1
        else:
            if cur_dup > 1:
                dup_runs.append(cur_dup)
            cur_dup = 1
        prev_fp = fp
    if cur_dup > 1:
        dup_runs.append(cur_dup)

    def _pct(vals, p):
        if not vals:
            return float("nan")
        s = sorted(vals)
        return s[min(int(p * len(s)), len(s) - 1)]

    n_frames = len(order)
    print(f"\nCSV sweep: {csv_path}")
    print(f"  {len(rows)} candidates across {n_frames} frames")
    if "foot_x" in rows[0]:
        print("  anchors from the foot_x column where set, bbox centers otherwise")
    else:
        print("  anchors fall back to bbox centers (no contour or foot_x column)")

    print(f"\n[MODES] {n_frames} frames:")
    for mode in ("two_boundary", "left_only", "right_only",
                 "single_uncalibrated", "none"):
        n = modes.get(mode, 0)
        flag = "   <-- never" if n == 0 else ""
        print(f" {mode:<22}{n:5}  ({100*n/max(n_frames,1):5.1f}%){flag}")

    print(f"\n[USABLE] boundaries per frame after gating:")
    for k in sorted(usable_counts):
        n = usable_counts[k]
        print(f" {k:<22}{n:5}  ({100*n/max(n_frames,1):5.1f}%)")

    print(f"\n[GATES] which gate rejected a candidate, {len(rows)} seen:")
    if not gate_hits:
        print("  nothing rejected")
    for gate, n in sorted(gate_hits.items(), key=lambda kv: -kv[1]):
        print(f" {gate:<22}{n:5}  ({100*n/max(len(rows),1):5.1f}%)")

    print(f"\n[FIELDS] observed range, for setting the gates from data:")
    for name, vals in field_values.items():
        print(f" {name:<16} min {min(vals):8.2f}  p10 {_pct(vals,0.10):8.2f}  "
              f"med {_pct(vals,0.50):8.2f}  p90 {_pct(vals,0.90):8.2f}  "
              f"max {max(vals):8.2f}")

    if offsets:
        print(f"\n[OFFSETS] {len(offsets)} frames produced a steering signal:")
        print(f" offset       min {min(offsets):+.3f}  med {_pct(offsets,0.5):+.3f}  "
              f"max {max(offsets):+.3f}")
        if widths:
            print(f" lane_width   min {min(widths):8.1f}  med {_pct(widths,0.5):8.1f}  "
                  f"max {max(widths):8.1f}")
    else:
        print("\n[OFFSETS] no frame produced a steering signal")

    print(f"\n[BLIND] longest run without a steering signal: "
          f"{blind_run} frames, starting at frame_id {blind_start}")

    # Identical consecutive frames. Expected when the robot is pushed by hand
    # and the scene barely changes, but a long run is also what a stalled
    # capture looks like, so it is reported either way rather than suppressed.
    if dup_runs:
        print(f"\n[DUPLICATE] {len(dup_runs)} runs of identical consecutive "
              f"frames, longest {max(dup_runs)}")
        if max(dup_runs) >= 5:
            print("            a run this long is also the signature of a "
                  "stalled capture — worth confirming against the source frames")
    else:
        print("\n[DUPLICATE] no identical consecutive frames")

    if video is not None:
        video.close()
        print(f"\n[VIDEO] {video.path}")
        print(f"  {video.frames_written} frames written: "
              f"{video_stats['source']} over source images, "
              f"{video_stats['blank']} on a blank canvas")
        if video_stats["unreadable"]:
            print(f"  {video_stats['unreadable']} matched images could not be read")
        if frame_index and video_stats["source"] == 0:
            print("  no frame_id matched a filename number — check the naming "
                  "against the CSV's frame_id column")
        print(f"  sidecar: {video.csv_path}")

    sys.exit(0 if passed == len(_results) else 1)