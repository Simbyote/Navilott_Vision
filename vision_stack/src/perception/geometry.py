"""Geometry branch: lane boundaries, stop lines and the stop sign, each from its own detector.

Purpose:
    Answers three structural questions per frame. Lanes and stop lines come
    from the gray lane ROI's Canny edges, computed once here and shared; the
    stop sign from the color sign ROI's redness, since a red sign on a floor
    of similar brightness gives Canny almost no edge. This module runs the
    three detectors and re-exports their names (phase2_perception.md, "Stage 3A").

Main package:
    GeometryBranchResult: one frame's accepted lane, stop-line and sign
    candidates, ROI-relative, with the frame identity.

Flow:
    1. Validate the ROIs.
    2. Canny on the lane ROI and the gradient split, once.
    3. Lanes (geometry_lanes), then stop lines (geometry_stop_line), from those edges.
    4. The sign (geometry_sign), from the color sign ROI.
    5. Package the three candidate lists.
"""
import time

import numpy as np

from src.perception.roi_crop import ROICropResult
# The detectors and what they share, each in its own module; every name is
# re-exported here so callers (and tests) import from geometry as before
from src.perception.geometry_shared import (  # noqa: F401
    CannyParams, GeometryBranchResult, GeometryConfig, LaneCandidate, LaneContourFilter, SignCandidate,
    SignContourFilter, StopLineCandidate, StopLineFilter, _canny, _close_edges, _contours, _horizontal_edges,
    _mean_contour_intensity, contour_foot_x,
)
from src.perception.geometry_lanes import (  # noqa: F401
    _extract_lane_candidates, _extreme_points, _filter_lane_contours, _lane_confidence, _merge_collinear,
    _strip_horizontal_lines, extract_lane_candidates, find_lane_candidates,
)
from src.perception.geometry_stop_line import (  # noqa: F401
    _EdgeSegment, _edge_segments, _fit_segment, _stop_line_confidence, _stop_line_from,
    extract_stop_line_candidates, find_stop_line_candidates,
)
from src.perception.geometry_sign import (  # noqa: F401
    TRACE_MIN_BBOX_FRAC, _extract_sign_candidates, _filter_sign_contours, _gate_sign, _letter_share,
    _red_mask, _redness, _shape_gate, _sign_candidate, _sign_confidence, _sign_polygon, _trace_entry,
    extract_sign_candidates, find_sign_candidates,
)


# The parts a geometry call times when given timings={}: the edges every
# lane-ROI detector reads (Canny and the gradient split), then each detector
GEOMETRY_PARTS = ("edges", "lanes", "stop_line", "sign")


class _PartClock:
    """Wall ms per part into timings, as each part ends; nothing at all when timings is None."""
    __slots__ = ("_timings", "_mark")

    def __init__(self, timings: dict | None) -> None:
        self._timings = timings
        self._mark = time.perf_counter() if timings is not None else 0.0

    def lap(self, name: str) -> None:
        if self._timings is not None:
            now = time.perf_counter()
            self._timings[name] = (now - self._mark) * 1000.0
            self._mark = now


def _splits(lane_roi, lane_edges, lane_filter: LaneContourFilter, stop_filter: StopLineFilter):
    """
    The gradient split each lane-ROI detector needs, computed once when their angles agree.

    Outputs:
        (stop-line split, lane split); the lane split is None when the lane
        filter is off or uses another angle (it then computes its own).
    """
    stop_split = _horizontal_edges(lane_roi, lane_edges, stop_filter.max_tilt_deg)
    lane_split = stop_split if lane_filter.horizontal_edge_deg == stop_filter.max_tilt_deg else None
    return stop_split, lane_split


def _validate_rois(where: str, lane_roi: np.ndarray, sign_color_roi: np.ndarray) -> None:
    """Raise, naming the ROI, unless lane_roi is (h, w) uint8 and sign_color_roi (h, w, 3) uint8."""
    for name, roi, ndim in (("lane_roi", lane_roi, 2), ("sign_color_roi", sign_color_roi, 3)):
        if roi is None:
            raise ValueError(f"{where}: {name} is None")
        if roi.dtype != np.uint8:
            raise TypeError(f"{where}: {name} expected uint8, got {roi.dtype}")
        if roi.ndim != ndim or (ndim == 3 and roi.shape[2] != 3):
            want = "single-channel grayscale (H,W)" if ndim == 2 else "BGR (H,W,3)"
            raise ValueError(f"{where}: {name} expected {want}, got {roi.shape}. "
                             "Color conversion belongs to preprocess.")


def run_geometry_branch(
    lane_roi: np.ndarray,
    sign_color_roi: np.ndarray,
    canny_params: CannyParams,
    lane_filter: LaneContourFilter,
    sign_filter: SignContourFilter,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    draw_overlays: bool = True,
    trace: bool = False,
    stop_filter: StopLineFilter = StopLineFilter(),
    timings: dict | None = None,
) -> tuple[GeometryBranchResult, dict, dict]:
    """
    Run lane, stop-line and sign detection on loose ROI arrays.

    Purpose:
        The entry point for tuning work. The pipeline calls run_geometry_stage().

    Inputs:
        lane_roi: (h, w) uint8 gray; sign_color_roi: (h, w, 3) uint8 BGR;
            e.g. from ROICropResult.
        canny_params: The lane ROI's Canny; the sign is found by color.
        draw_overlays: Build the debug overlays in both branches. False skips
            four full-ROI allocations and two contour rasterizations per frame.
        trace: Record the per-contour sign trace (see extract_sign_candidates)
            and the lane contour trace (see extract_lane_candidates).
        stop_filter: Stop-line gates. Stop lines read the lane ROI's Canny
            edges and never change the lane or sign results.
        timings: A dict to fill with each GEOMETRY_PARTS' wall ms; None times nothing.

    Outputs:
        (result, lane_debug, sign_debug). lane_debug["stop_line"] holds the
        stop-line detector's debug (see extract_stop_line_candidates).

    Raises:
        ValueError / TypeError: If either ROI is None, not uint8, or the
            wrong shape. The message names the offending ROI.
    """
    _validate_rois("run_geometry_branch", lane_roi, sign_color_roi)
    clock = _PartClock(timings)

    lane_edges = _canny(lane_roi, canny_params)     # shared by lanes and stop lines
    stop_split, lane_split = _splits(lane_roi, lane_edges, lane_filter, stop_filter)
    clock.lap("edges")

    lane_candidates, lane_debug = extract_lane_candidates(
        lane_roi,
        canny_params,
        lane_filter,
        frame_id,
        timestamp_ms,
        draw_overlays,
        edges_raw = lane_edges,
        horizontal_split = lane_split,
        trace = trace,
    )
    clock.lap("lanes")

    stop_line_candidates, lane_debug["stop_line"] = extract_stop_line_candidates(
        lane_roi,
        lane_edges,
        stop_filter,
        frame_id,
        timestamp_ms,
        split = stop_split,
    )
    clock.lap("stop_line")

    sign_candidates, sign_debug = extract_sign_candidates(
        sign_color_roi,
        sign_filter,
        frame_id,
        timestamp_ms,
        draw_overlays,
        trace
    )
    clock.lap("sign")

    result = GeometryBranchResult(
        lane_candidates,
        sign_candidates,
        frame_id,
        timestamp_ms,
        stop_line_candidates,
    )

    return result, lane_debug, sign_debug


def run_geometry_stage(
        roi: ROICropResult,
        config: GeometryConfig = GeometryConfig(),
        draw_overlays: bool = False,
        trace: bool = False,
        timings: dict | None = None,
    ) -> tuple[GeometryBranchResult, dict, dict]:
    """
    Stage entry point: run the geometry branch on one ROICropResult.

    Purpose:
        Same shape as preprocess_frame() and crop_rois(), so the orchestrator
        chains single-argument calls instead of re-threading the frame
        identity by hand. A thin adapter over run_geometry_branch().

    Inputs:
        draw_overlays: Off by default; the live loop discards the debug dicts,
            so building them would cost allocations nothing reads. The
            dataset harness passes True.
        trace: Off by default. Records the sign trace for the debug views.
        timings: A dict to fill with each GEOMETRY_PARTS' wall ms; None times nothing.

    Outputs:
        (result, lane_debug, sign_debug), with frame_id and timestamp_ms
        carried from the ROICropResult.
    """
    return run_geometry_branch(
        lane_roi = roi.lane_roi,
        sign_color_roi = roi.sign_color_roi,
        canny_params = config.canny,
        lane_filter = config.lane,
        sign_filter = config.sign,
        frame_id = roi.frame_id,
        timestamp_ms = roi.timestamp_ms,
        draw_overlays = draw_overlays,
        trace = trace,
        stop_filter = config.stop_line,
        timings = timings,
    )


def detect_geometry(
        roi: ROICropResult,
        config: GeometryConfig = GeometryConfig(),
        timings: dict | None = None,
    ) -> GeometryBranchResult:
    """
    Production twin of run_geometry_stage(): same stage, result only.

    Inputs:
        timings: A dict to fill with each GEOMETRY_PARTS' wall ms; None times nothing.

    Outputs:
        GeometryBranchResult, with frame_id and timestamp_ms carried from roi.

    Raises:
        ValueError / TypeError: If either ROI is None, not uint8, or the
            wrong shape. The message names the offending ROI.
    """
    _validate_rois("detect_geometry", roi.lane_roi, roi.sign_color_roi)
    clock = _PartClock(timings)

    lane_edges = _canny(roi.lane_roi, config.canny)     # shared by lanes and stop lines
    stop_split, lane_split = _splits(roi.lane_roi, lane_edges, config.lane, config.stop_line)
    clock.lap("edges")

    lane_candidates = find_lane_candidates(
        roi.lane_roi,
        config.canny,
        config.lane,
        roi.frame_id,
        roi.timestamp_ms,
        edges_raw = lane_edges,
        horizontal_split = lane_split,
    )
    clock.lap("lanes")

    stop_line_candidates = find_stop_line_candidates(
        roi.lane_roi,
        lane_edges,
        config.stop_line,
        roi.frame_id,
        roi.timestamp_ms,
        split = stop_split,
    )
    clock.lap("stop_line")

    sign_candidates = find_sign_candidates(
        roi.sign_color_roi,
        config.sign,
        roi.frame_id,
        roi.timestamp_ms,
    )
    clock.lap("sign")

    return GeometryBranchResult(
        lane_candidates,
        sign_candidates,
        roi.frame_id,
        roi.timestamp_ms,
        stop_line_candidates,
    )
