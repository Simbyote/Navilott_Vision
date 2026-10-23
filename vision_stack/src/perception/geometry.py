"""Geometry branch: lane boundaries and stop lines from the gray lane ROI, stop-sign shapes from the color sign ROI.

Purpose:
    Answers three structural questions per frame. Lane boundaries: white tape
    on a dark mat gives intensity edges that Canny finds, and contours are
    kept when their area, elongation, span and brightness match tape. Stop
    lines: the same lane-ROI Canny edges, split by gradient direction so only
    edges of near-horizontal lines remain, then the tape's top and bottom
    edges paired into one line with its own gates. Stop sign: a red sign on a
    floor of similar brightness gives Canny almost no edge in gray, so the
    sign is found by color instead: a redness mask, its largest blob, and
    polygon approximation of that blob's hull, where an octagon gives about 8
    vertices and area and solidity reject noise.

    Stop lines only share the Canny result. They have their own config
    (StopLineFilter) and never change what the lane and sign detectors see,
    so lane and sign results are identical with or without them. Geometry
    only detects; the distance to a stop line is stop_line_distance's job.

    This module is the orchestrator: it validates the ROIs, computes what
    the detectors share once, runs each and packages the result. Each
    detector lives in its own module, and what they share in another:

        geometry_lanes.py       lane boundaries (find_lane_candidates)
        geometry_stop_line.py   stop lines (find_stop_line_candidates)
        geometry_sign.py        the stop sign (find_sign_candidates)
        geometry_shared.py      configs, candidate types, edge helpers

    Every name from them is re-exported here, so callers import from
    geometry as they always have.

Main package:
    detect_geometry(): production, one frame's result.
    run_geometry_branch() / run_geometry_stage(): the same with traces, for
        the debug views and tests.
    GeometryBranchResult: one frame's accepted LaneCandidates,
    SignCandidates and StopLineCandidates, all coordinates ROI-relative, with
    the frame identity carried from ROICropResult. Consumed by feature
    fusion, lane_offset and stop_line_distance.

Flow:
    1. Validate the ROIs: the lane ROI single-channel, the sign ROI BGR, both uint8.
    2. Canny on the lane ROI, once; lanes and stop lines both read it.
    3. Lane: take lines lying across the lane out of the edges (the stop
       line, so it can't close into one contour with the lane lines), close
       along-line gaps, filter contours, merge fragments.
    4. Stop line: keep near-horizontal edges, split them into top (dark to
       bright going down) and bottom edges, fit each, pair top with bottom,
       gate by length, tilt, thickness and brightness.
    5. Sign: redness mask of the sign ROI (Otsu, floored), close, then gate
       the blobs largest first (area, vertex count, solidity, the ghost-sign
       gates) and take the first that passes: a larger red thing in the ROI
       (a chair, a box) refused no longer hides the sign behind it.
    6. Package the three candidate lists with the frame identity.
"""
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

    Outputs:
        (result, lane_debug, sign_debug). lane_debug["stop_line"] holds the
        stop-line detector's debug (see extract_stop_line_candidates).

    Raises:
        ValueError / TypeError: If either ROI is None, not uint8, or the
            wrong shape. The message names the offending ROI.
    """
    _validate_rois("run_geometry_branch", lane_roi, sign_color_roi)

    lane_edges = _canny(lane_roi, canny_params)     # shared by lanes and stop lines
    stop_split, lane_split = _splits(lane_roi, lane_edges, lane_filter, stop_filter)

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

    stop_line_candidates, lane_debug["stop_line"] = extract_stop_line_candidates(
        lane_roi,
        lane_edges,
        stop_filter,
        frame_id,
        timestamp_ms,
        split = stop_split,
    )

    sign_candidates, sign_debug = extract_sign_candidates(
        sign_color_roi,
        sign_filter,
        frame_id,
        timestamp_ms,
        draw_overlays,
        trace
    )

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
    )


def detect_geometry(
        roi: ROICropResult,
        config: GeometryConfig = GeometryConfig(),
    ) -> GeometryBranchResult:
    """
    Production twin of run_geometry_stage(): same stage, result only.

    Outputs:
        GeometryBranchResult, with frame_id and timestamp_ms carried from roi.

    Raises:
        ValueError / TypeError: If either ROI is None, not uint8, or the
            wrong shape. The message names the offending ROI.
    """
    _validate_rois("detect_geometry", roi.lane_roi, roi.sign_color_roi)

    lane_edges = _canny(roi.lane_roi, config.canny)     # shared by lanes and stop lines
    stop_split, lane_split = _splits(roi.lane_roi, lane_edges, config.lane, config.stop_line)

    lane_candidates = find_lane_candidates(
        roi.lane_roi,
        config.canny,
        config.lane,
        roi.frame_id,
        roi.timestamp_ms,
        edges_raw = lane_edges,
        horizontal_split = lane_split,
    )

    stop_line_candidates = find_stop_line_candidates(
        roi.lane_roi,
        lane_edges,
        config.stop_line,
        roi.frame_id,
        roi.timestamp_ms,
        split = stop_split,
    )

    sign_candidates = find_sign_candidates(
        roi.sign_color_roi,
        config.sign,
        roi.frame_id,
        roi.timestamp_ms,
    )

    return GeometryBranchResult(
        lane_candidates,
        sign_candidates,
        roi.frame_id,
        roi.timestamp_ms,
        stop_line_candidates,
    )
