"""Lane offset: the robot's lateral error from the center of its own lane.

Purpose:
    Turns geometry's lane candidates into the steering error Phase 3 uses.
    It reads the candidates, not fusion's output, because it needs each
    contour's foot, width and brightness, not just a centroid.

Main package:
    LaneOffsetResult: offset in [-1, 1] of half the lane ROI width, + = robot
    right of center, with the mode it came from.
"""
import numpy as np
from dataclasses import dataclass

from src.params import (
    FOOT_BAND_PX, MODE_LEFT_ONLY, MODE_NONE, MODE_RIGHT_ONLY,
    MODE_SINGLE_UNCALIBRATED, MODE_TWO_BOUNDARY,
)
from src.utils import check_same_frame, clamp
from src.perception.geometry import GeometryBranchResult, LaneCandidate, StopLineCandidate
from src.perception.roi_crop import ROICropResult


@dataclass(frozen=True)
class LaneOffsetConfig:
    """Stricter than geometry's gates: geometry asks whether it's a marking, this whether to steer by it."""
    conf_threshold: float = 0.30        # min candidate confidence
    min_proximity: float = 0.25         # higher up the ROI says little about where the robot is now
    min_length_px: float = 25.0         # keeps dash fragments out
    min_width_px: float = 1.0
    max_width_px: float = 25.0          # wider is a blob, glare or a merged pair
    min_intensity: float = 90.0         # rejects shadow edges and mat seams
    min_lane_width_px: float = 60.0     # closer anchors are fragments of one marking
    max_lane_width_px: float = 400.0    # wider apart spans more than one lane
    # Lane-ROI px from a boundary to the lane center; None disables one-boundary
    # mode. MEASURED sets the course's 178 (config.py)
    expected_half_lane_px: float | None = 228
    # Lane-ROI px of the robot's centerline: where a centered robot sees its lane's middle.
    # None is the ROI's middle, right for a camera square on the centerline
    robot_x_px: float | None = None
    foot_band_px: int = FOOT_BAND_PX    # only when geometry didn't store a foot_x
    # Candidates this much within a stop line are it, or the pocket it encloses: skipped. None disables
    stop_line_overlap: float | None = 0.5


@dataclass(frozen=True)
class BoundaryAnchor:
    foot_x: float               # lane-ROI px, from the contour's lowest rows
    weight: float               # confidence scaled by proximity
    candidate: LaneCandidate


@dataclass(frozen=True)
class LaneOffsetResult:
    """boundary_count counts usable boundaries, so 0 separates seeing nothing from nothing to steer by."""
    offset: float                   # [-1, 1] of half the lane ROI width; + = robot right of lane center
    left_x: float | None            # left anchor x, lane-ROI px; None if unused
    right_x: float | None           # right anchor x, lane-ROI px; None if unused
    lane_width_px: float | None     # right_x - left_x; None in single-sided modes
    confidence: float               # mean weight of the anchors actually used
    boundary_count: int
    mode: str                       # "two_boundary" | "left_only" | "right_only" | "single_uncalibrated" | "none"
    frame_id: int
    timestamp_ms: int
    other_side_offset: float | None = None  # one boundary: the offset were it the lane's other side


def foot_x(
        candidate: LaneCandidate,
        band_px: int = FOOT_BAND_PX
    ) -> float:
    """
    Where the marking meets the near edge, in lane-ROI px: the stored foot_x, else the
    contour's lowest band, else the bbox center. An angled line's centroid sits halfway up the ROI.
    """
    # Also covers CSV replays, whose candidates have no contour
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
    """Logs the gate that rejected it, so the summary says why a frame went blind."""
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


def _is_usable(
        candidate: LaneCandidate,
        config: LaneOffsetConfig,
    ) -> bool:
    """Production twin of _usable(), without the log."""
    c = candidate
    if c.confidence < config.conf_threshold:
        return False
    if c.proximity < config.min_proximity:
        return False
    if c.length_px < config.min_length_px:
        return False
    if not (config.min_width_px <= c.width_px <= config.max_width_px):
        return False
    if c.mean_intensity < config.min_intensity:
        return False
    return True


def _anchor(
        candidate: LaneCandidate,
        config: LaneOffsetConfig,
    ) -> BoundaryAnchor:
    # A strong marking high in the ROI is real; it just says less about now
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
    (left, right), either may be None: the nearest usable anchor on each side of center,
    not the outermost pair, which on a two-lane street would center the robot on the
    street. All on one side (drifted out): the two nearest center. An anchor at center
    counts as right.
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
        scale: float,
        config: LaneOffsetConfig,
        frame_id: int,
        timestamp_ms: int,
        boundary_count: int,
        log: list[str],
    ) -> LaneOffsetResult:
    """
    Offset from one boundary, projected to the lane center so it's the same quantity as
    two-boundary mode. Without expected_half_lane_px it reports single_uncalibrated at
    zero confidence instead of an error of unknown scale.
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

    offset = clamp((center_x - implied_center) / scale, -1.0, 1.0)
    # The side is a guess from the ROI half; past a boundary it's wrong, and
    # Phase 3 takes this instead when the guess breaks continuity
    mirrored = implied_center + (2 * half if side == "right" else -2 * half)
    other_side = clamp((center_x - mirrored) / scale, -1.0, 1.0)
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
        other_side_offset = round(other_side, 4),
    )


def _project_single(
        anchor: BoundaryAnchor,
        side: str,
        center_x: float,
        scale: float,
        config: LaneOffsetConfig,
        frame_id: int,
        timestamp_ms: int,
        boundary_count: int,
    ) -> LaneOffsetResult:
    """Production twin of _single_sided(), without the log line."""
    half = config.expected_half_lane_px
    if half is None:
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

    offset = clamp((center_x - implied_center) / scale, -1.0, 1.0)
    mirrored = implied_center + (2 * half if side == "right" else -2 * half)
    other_side = clamp((center_x - mirrored) / scale, -1.0, 1.0)
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
        other_side_offset = round(other_side, 4),
    )


def _on_stop_line(
        candidate: LaneCandidate,
        stop_lines: list[StopLineCandidate],
        min_overlap: float | None,
    ) -> StopLineCandidate | None:
    """
    The stop line this candidate belongs to, or None: lying across the ROI, overlapping
    the line's span and within its thickness of it. A lane line crossing it runs along the ROI and stays.
    """
    if min_overlap is None or not stop_lines:
        return None
    x, y, w, h = candidate.bbox
    if w < h:
        return None
    for s in stop_lines:
        sx, sy, sw, sh = s.bbox
        x_overlap = min(x + w, sx + sw) - max(x, sx)
        rows_apart = max(sy - (y + h), y - (sy + sh))       # <= 0 when the boxes share rows
        if x_overlap >= min_overlap * w and rows_apart <= s.thickness_px:
            return s
    return None


def compute_lane_offset(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
        config: LaneOffsetConfig = LaneOffsetConfig(),
    ) -> tuple[LaneOffsetResult, dict]:
    """
    (result, debug summary). Anchors in lane-ROI px. ValueError if an input is None or the frames disagree.
    """
    check_same_frame(geometry, roi, "compute_lane_offset")

    log = []
    frame_id = geometry.frame_id
    timestamp_ms = geometry.timestamp_ms

    roi_width = roi.lane_rect[2]
    scale = roi_width / 2.0                 # [-1, 1] spans half the ROI, wherever the robot sits in it
    center_x = scale if config.robot_x_px is None else float(config.robot_x_px)

    raw = list(geometry.lane_candidates)
    usable = []
    for c in raw:
        on = _on_stop_line(c, geometry.stop_line_candidates, config.stop_line_overlap)
        if on is not None:
            log.append(f"[REJECT] candidate at x={c.bbox[0]} y={c.bbox[1]} lies on the "
                       f"stop line at y={on.y_near_px:.1f}")
        elif _usable(c, config, log):
            usable.append(c)
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
        # Implausible spacing falls back to the strongest single anchor
        lane_width_px = right.foot_x - left.foot_x

        if lane_width_px < config.min_lane_width_px:
            log.append(
                f"[MERGE] anchors {lane_width_px:.1f}px apart, below "
                f"{config.min_lane_width_px} — same marking, not opposite boundaries"
            )
            best = max(anchors, key=lambda a: a.weight)
            side = "left" if best.foot_x < center_x else "right"
            return _summary(_single_sided(best, side, center_x, scale, config,
                                          frame_id, timestamp_ms,
                                          boundary_count, log))

        if lane_width_px > config.max_lane_width_px:
            log.append(
                f"[SPAN] anchors {lane_width_px:.1f}px apart, above "
                f"{config.max_lane_width_px} — pair spans more than one lane"
            )
            best = max(anchors, key=lambda a: a.weight)
            side = "left" if best.foot_x < center_x else "right"
            return _summary(_single_sided(best, side, center_x, scale, config,
                                          frame_id, timestamp_ms,
                                          boundary_count, log))

        lane_center = (left.foot_x + right.foot_x) / 2.0
        offset = clamp((center_x - lane_center) / scale, -1.0, 1.0)
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
    return _summary(_single_sided(anchor, side, center_x, scale, config,
                                  frame_id, timestamp_ms, boundary_count, log))


def estimate_lane_offset(
        geometry: GeometryBranchResult,
        roi: ROICropResult,
        config: LaneOffsetConfig = LaneOffsetConfig(),
    ) -> LaneOffsetResult:
    """Production twin of compute_lane_offset(), without the debug summary."""
    check_same_frame(geometry, roi, "estimate_lane_offset")

    frame_id = geometry.frame_id
    timestamp_ms = geometry.timestamp_ms

    roi_width = roi.lane_rect[2]
    scale = roi_width / 2.0                 # [-1, 1] spans half the ROI, wherever the robot sits in it
    center_x = scale if config.robot_x_px is None else float(config.robot_x_px)

    usable = [c for c in geometry.lane_candidates
              if _on_stop_line(c, geometry.stop_line_candidates, config.stop_line_overlap) is None
              and _is_usable(c, config)]
    anchors = [_anchor(c, config) for c in usable]
    boundary_count = len(anchors)

    if not anchors:
        return LaneOffsetResult(
            offset = 0.0, left_x = None, right_x = None, lane_width_px = None,
            confidence = 0.0, boundary_count = 0, mode = MODE_NONE,
            frame_id = frame_id, timestamp_ms = timestamp_ms,
        )

    left, right = _lane_pair(anchors, center_x)

    if left is not None and right is not None:
        # Implausible spacing falls back to the strongest single anchor
        lane_width_px = right.foot_x - left.foot_x

        if (lane_width_px < config.min_lane_width_px
                or lane_width_px > config.max_lane_width_px):
            best = max(anchors, key=lambda a: a.weight)
            side = "left" if best.foot_x < center_x else "right"
            return _project_single(best, side, center_x, scale, config,
                                   frame_id, timestamp_ms, boundary_count)

        lane_center = (left.foot_x + right.foot_x) / 2.0
        offset = clamp((center_x - lane_center) / scale, -1.0, 1.0)
        total_weight = left.weight + right.weight
        mean_weight = total_weight / 2.0 if total_weight > 0 else 0.0

        return LaneOffsetResult(
            offset = round(offset, 4),
            left_x = round(left.foot_x, 2),
            right_x = round(right.foot_x, 2),
            lane_width_px = round(lane_width_px, 2),
            confidence = round(mean_weight, 4),
            boundary_count = boundary_count,
            mode = MODE_TWO_BOUNDARY,
            frame_id = frame_id,
            timestamp_ms = timestamp_ms,
        )

    anchor = left if left is not None else right
    side = "left" if left is not None else "right"
    return _project_single(anchor, side, center_x, scale, config,
                           frame_id, timestamp_ms, boundary_count)