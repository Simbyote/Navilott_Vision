"""
phase2_linker.py

Phase 1 and 2 Pipeline Linker, excluding sign and traffic integration

Purpose:
    Defines the stage order once, in run_chain():

        FrameData -> preprocess_frame -> crop_rois -> run_geometry_stage
                  -> run_color_stage -> compute_lane_offset
                  -> fuse_detections -> package_phase2

    The last stage's Phase2Output is the Phase 2 -> Phase 3 handoff. The
    color branch feeds fusion here, not the other way round: fusion is where
    its traffic-light candidates meet the geometry branch's lane and sign
    candidates, so the linker, which owns the stage order, runs it and hands
    the result on.

    The color branch needs calibrated HSV ranges. PipelineConfig.color leaves
    it OFF until they are supplied (see load_color_config), in which case
    fusion receives no traffic candidates and Phase2Output has no
    traffic_light detection. Lane boundaries and stop signs flow either way.

    Everything else that needs the chain goes through here so nothing can
    drift from it: the live view (run_live_view), and the tests under test/.

Live view:
    run_live_view() hands run_chain() to live_view.run() as a plain callable.
    live_view never imports this module, so the dependency runs one way:

        debug_video  <-  live_view  <-  phase2_linker

    Command line (all options come from live_view.cli):
        python3 phase2_linker.py --video clip.mp4 --no-display
        python3 phase2_linker.py --frames frames/Sample1

Tests:
    Ground truth, metamorphic properties and replay live in test/. The
    synthetic_frame() and expected_offset() fixtures below are kept here only
    until those tests move; nothing in this module uses them.

Configuration:
    Every stage's tuning lives in PipelineConfig. A camera change moves the
    cm-per-pixel scale, the ROI bounds and the gate thresholds together, so
    re-tuning after a hardware change is meant to be a config swap and a
    re-run, not a code edit.

What this cannot test:
    Accuracy in centimetres against the +/-2cm requirement. That needs
    captures at measured lateral offsets on a real course. Synthetic ground
    truth proves the arithmetic recovers what was drawn; it says nothing
    about whether the camera sees the world the way the synthetic frames
    assume.
"""
import time

import numpy as np
import cv2
from dataclasses import dataclass, field, replace

from src.capture.camera import FrameData
from src.perception.preprocess import preprocess_frame, PreprocessParams
from src.perception.roi_crop import crop_rois, ROIConfig
from src.perception.geometry import run_geometry_stage, GeometryConfig
from src.perception.color_branch import ColorConfig, run_color_stage, load_hsv_ranges
from src.perception.lane_offset import compute_lane_offset, LaneOffsetConfig
from src.perception.feature_fusion import fuse_detections
from src.perception.phase2_out import package_phase2
import src.debugger.live_view as live_view

# =============================================================================
# Pipeline Configuration
# =============================================================================
@dataclass(frozen=True)
class PipelineConfig:
    """
    Every stage's tuning as one unit

    preprocess: conditioning parameters
    roi: ROI bounds
    geometry: contour filters and edge detection
    color: traffic-light HSV ranges and blob filter. Off until ranges are given
    lane_offset: boundary gates and the calibration constant
    """
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    roi: ROIConfig = field(default_factory=ROIConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    lane_offset: LaneOffsetConfig = field(default_factory=LaneOffsetConfig)

# Gates revised from the 4827-candidate CSV sweep rather than from course
# dimensions. min_proximity sat above the observed median of 0.17 and killed
# 1752 candidates the geometry branch had already scored at or above 0.30;
# max_width_px at 25 clipped detections whose p90 is 27 and max is 41.7.
MEASURED = PipelineConfig(
    lane_offset = LaneOffsetConfig(
        conf_threshold = 0.25,
        min_proximity = 0.05,
        max_width_px = 45.0,
        min_intensity = 130.0,
    )
)

# =============================================================================
# Chain Result
# =============================================================================
@dataclass(frozen=True)
class ChainResult:
    """
    Every stage's output for one frame, so a failure can be traced to the
    stage that produced it rather than only to the final number

    Stop signs travel in geometry.sign_candidates. sign_debug carries the
    sign branch's reject_counts and edge map, plus the per-contour trace when
    run_chain was asked for it

    fusion, fusion_debug: FusionResult and its debug summary
    phase2: the Phase2Output handed to Phase 3
    timings_ms: wall time of each stage in ms, keyed preprocess, roi,
                geometry, color, lane_offset, fusion, package
    traffic: TrafficLightCandidates from the color branch ([] when it is off)
    traffic_debug: the color branch's masks, counts and trace; just
                   {"enabled": False} when it is off
    """
    frame: FrameData
    pre: object
    roi: object
    geometry: object
    offset: object
    lane_debug: dict
    offset_debug: dict
    sign_debug: dict = field(default_factory=dict)
    fusion: object = None
    fusion_debug: dict = field(default_factory=dict)
    phase2: object = None
    timings_ms: dict = field(default_factory=dict)
    traffic: list = field(default_factory=list)
    traffic_debug: dict = field(default_factory=dict)

# =============================================================================
# The Chain
# =============================================================================
def run_chain(
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: int = 0,
        config: PipelineConfig = PipelineConfig(),
        draw_overlays: bool = False,
        trace: bool = False,
    ) -> ChainResult:
    """
    Purpose:
        Run one frame through every stage. This is the only place the stage
        order is written down; the harnesses below and the live loop should
        both come through here so they cannot drift

    Inputs:
        frame_bgr: (H, W, 3) uint8 BGR frame as CameraSource.read() delivers it
        frame_id, timestamp_ms: the stamp capture would have assigned
        config: PipelineConfig
        draw_overlays: build the geometry debug overlays
        trace: record the per-contour sign trace and the per-blob traffic
               trace in the debug dicts

    Outputs:
        ChainResult
    """
    timings = {}
    mark = [time.perf_counter()]

    def lap(name):
        now = time.perf_counter()
        timings[name] = (now - mark[0]) * 1000.0
        mark[0] = now

    fd = FrameData(frame_bgr, frame_id, timestamp_ms)

    pre = preprocess_frame(fd, config.preprocess)
    lap("preprocess")

    roi = crop_rois(pre, config.roi)
    lap("roi")

    geo, lane_debug, sign_debug = run_geometry_stage(
        roi, config.geometry, draw_overlays, trace
    )
    lap("geometry")

    traffic, traffic_debug = run_color_stage(roi, roi.traffic_roi, config.color, trace)
    lap("color")

    offset, offset_debug = compute_lane_offset(geo, roi, config.lane_offset)
    lap("lane_offset")

    fusion, fusion_debug = fuse_detections(geo, traffic, roi)
    lap("fusion")

    phase2 = package_phase2(fusion, offset)
    lap("package")

    return ChainResult(fd, pre, roi, geo, offset, lane_debug, offset_debug,
                       sign_debug, fusion, fusion_debug, phase2, timings,
                       traffic, traffic_debug)

# =============================================================================
# Live View
# =============================================================================
def run_live_view(source, config: PipelineConfig = MEASURED, trace: bool = True,
                  hsv_path=None, **options):
    """
    Purpose:
        Run a live_view frame source through run_chain() with the debug
        overlay, optional window and on-disk recording

    Inputs:
        source: a live_view.FrameSource (camera, video file or image directory)
        config: PipelineConfig; drives both the chain and the overlay gating
        trace: record the per-contour sign trace and per-blob traffic trace
               so the views can show rejected candidates. On by default here
               because this is the debug entry point; run_chain leaves it off
        hsv_path: calibrated HSV ranges JSON. Switches the color branch on for
               this run; None keeps whatever config.color says
        options: out_dir, display, scale, stride, limit, fps, views, passed
                 to live_view.run()

    Outputs:
        live_view.RunStats
    """
    if hsv_path:
        config = replace(config, color=ColorConfig(load_hsv_ranges(hsv_path),
                                                   config.color.blob))

    def process(frame_bgr, frame_id, timestamp_ms):
        return run_chain(frame_bgr, frame_id, timestamp_ms, config, trace=trace)

    return live_view.run(source, process, config.lane_offset, **options)

# =============================================================================
# Synthetic Frame Construction
# =============================================================================
FRAME_H, FRAME_W = 270, 480
LANE_RECT = (24, 252, 432, 108)          # crop_rois output at 480x360
ROI_W, ROI_H = LANE_RECT[2], LANE_RECT[3]
ROI_CENTER = ROI_W / 2.0                 # 216.0

def synthetic_frame(
        marks,
        mark_width: int = 6,
        road: int = 60,
        surround: int = 30,
        marking: int = 240,
    ) -> np.ndarray:
    """
    Purpose:
        Build a BGR frame whose lane ROI contains markings at known
        ROI-local x positions, so the correct lane offset is known exactly

    Inputs:
        marks: iterable of ROI-local x positions, or of
               (x, y_top, y_bottom) to control vertical extent for dash and
               partial-visibility cases
        mark_width: marking width in px
        road, surround, marking: intensities for the road surface inside the
               lane ROI, everything outside it, and the markings

    Outputs:
        (360, 480, 3) uint8 BGR

    Notes:
        A marking drawn at ROI x is recovered by the chain at ROI x. Verified
        to the pixel: drawn at 150 and 290, detected at 150.0 and 290.0
    """
    x0, y0, w, h = LANE_RECT
    frame = np.full((FRAME_H, FRAME_W, 3), surround, np.uint8)
    frame[y0:y0 + h, x0:x0 + w] = road

    for mark in marks:
        if isinstance(mark, (int, float)):
            x, top, bottom = mark, 0, h
        else:
            x, top, bottom = mark
        fx = x0 + int(x)
        cv2.rectangle(
            frame,
            (fx - mark_width // 2, y0 + int(top)),
            (fx + mark_width // 2, y0 + int(bottom) - 1),
            (marking,) * 3, -1,
        )
    return frame

def expected_offset(left_x: float, right_x: float) -> float:
    """The offset the chain must recover for markings at these ROI x."""
    lane_center = (left_x + right_x) / 2.0
    return (ROI_CENTER - lane_center) / ROI_CENTER

if __name__ == "__main__":
    import sys
    sys.exit(live_view.cli(run_live_view))