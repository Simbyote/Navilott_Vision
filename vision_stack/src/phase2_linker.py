"""Phase 1-2 linker: the one place the stage order is written down.

Purpose:
    Runs a frame through every Phase 1-2 stage in order, so the live view and
    the tests all go through one chain and can't drift from it. The color
    branch stays off until calibrated HSV ranges are supplied; lane
    boundaries and stop signs flow either way. Every stage's tuning lives in
    PipelineConfig, so re-tuning after a camera change is a config swap and a
    re-run, not a code edit.

Main package:
    ChainResult: every stage's output and debug data for one frame, ending in
    the Phase2Output handed to Phase 3, plus per-stage timings.

Flow:
    FrameData -> preprocess_frame -> crop_rois -> run_geometry_stage
              -> run_color_stage -> compute_lane_offset
              -> fuse_detections -> package_phase2

Command line (options from live_view.cli):
    python3 phase2_linker.py --video clip.mp4 --no-display
    python3 phase2_linker.py --frames frames/Sample1
"""
import time

import numpy as np
import cv2
from dataclasses import dataclass, field, replace

from src.capture.camera import FrameData
from src.params import FRAME_H, FRAME_W
from src.perception.preprocess import preprocess_frame, PreprocessParams, PreprocessResult
from src.perception.roi_crop import crop_rois, ROIConfig, ROICropResult, LANE, resolve
from src.perception.geometry import run_geometry_stage, GeometryConfig, GeometryBranchResult
from src.perception.color_branch import ColorConfig, run_color_stage, load_hsv_ranges
from src.perception.lane_offset import compute_lane_offset, LaneOffsetConfig, LaneOffsetResult
from src.perception.feature_fusion import fuse_detections, FusionResult
from src.perception.phase2_out import package_phase2, Phase2Output
import src.debugger.live_view as live_view


@dataclass(frozen=True)
class PipelineConfig:
    """Every stage's tuning as one unit."""
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    roi: ROIConfig = field(default_factory=ROIConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    color: ColorConfig = field(default_factory=ColorConfig)                 # off until HSV ranges are given
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


@dataclass(frozen=True)
class ChainResult:
    """
    Every stage's output for one frame, so a failure can be traced to the
    stage that produced it rather than only to the final number.
    """
    frame: FrameData
    pre: PreprocessResult
    roi: ROICropResult
    geometry: GeometryBranchResult          # stop signs travel in its sign_candidates
    offset: LaneOffsetResult
    lane_debug: dict                        # geometry's lane debug dict
    offset_debug: dict                      # compute_lane_offset's debug summary
    sign_debug: dict = field(default_factory=dict)      # reject_counts and edge map, plus the per-contour trace when requested
    fusion: FusionResult | None = None
    fusion_debug: dict = field(default_factory=dict)
    phase2: Phase2Output | None = None      # the Phase 3 handoff
    timings_ms: dict = field(default_factory=dict)      # wall ms per stage: preprocess, roi, geometry, color, lane_offset, fusion, package
    traffic: list = field(default_factory=list)         # TrafficLightCandidates; [] when the color branch is off
    traffic_debug: dict = field(default_factory=dict)   # masks, counts and trace; just {"enabled": False} when off


def run_chain(
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: int = 0,
        config: PipelineConfig = PipelineConfig(),
        draw_overlays: bool = False,
        trace: bool = False,
    ) -> ChainResult:
    """
    Run one frame through every Phase 1-2 stage.

    Inputs:
        frame_bgr: (H, W, 3) uint8 BGR, as CameraSource.read() delivers it.
        frame_id, timestamp_ms: The stamp capture would have assigned.
        config: Defaults to PipelineConfig(), the shipped tuning.
            run_live_view() defaults to MEASURED instead.
        draw_overlays: Build the geometry debug overlays.
        trace: Record the per-contour sign trace and the per-blob traffic
            trace in the debug dicts.

    Outputs:
        ChainResult, with each stage's wall time in timings_ms.
    """
    timings = {}
    mark = [time.perf_counter()]    # a list so lap() can update it without nonlocal

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

    traffic, traffic_debug = run_color_stage(roi, config.color, trace)
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


def run_live_view(source, config: PipelineConfig = MEASURED, trace: bool = True,
                  hsv_path: str | None = None, **options) -> "live_view.RunStats":
    """
    Run a live_view frame source through run_chain() with the debug overlay.

    Purpose:
        Hands run_chain() to live_view.run() as a plain callable. live_view
        never imports this module, so the dependency runs one way:

            debug_video  <-  live_view  <-  phase2_linker

    Inputs:
        source: A live_view.FrameSource: camera, video file or image directory.
        config: Drives both the chain and the overlay gating. Defaults to MEASURED.
        trace: On by default, since this is the debug entry point; run_chain()
            leaves it off.
        hsv_path: Calibrated HSV ranges JSON. Switches the color branch on for
            this run; None keeps whatever config.color says.
        options: out_dir, display, scale, stride, limit, fps, views; passed
            to live_view.run().

    Outputs:
        live_view.RunStats.

    Side effects:
        Reads hsv_path. Whatever live_view.run() does with the options:
        display windows and recordings under out_dir.
    """
    if hsv_path:
        config = replace(config, color=ColorConfig(load_hsv_ranges(hsv_path),
                                                   config.color.blob))

    def process(frame_bgr, frame_id, timestamp_ms):
        return run_chain(frame_bgr, frame_id, timestamp_ms, config, trace=trace)

    return live_view.run(source, process, config.lane_offset, **options)


# @TODO move synthetic_frame() and expected_offset() into test/; nothing in this module uses them
LANE_RECT = resolve(LANE, (FRAME_H, FRAME_W))   # what crop_rois will cut at this size
ROI_W, ROI_H = LANE_RECT[2], LANE_RECT[3]
ROI_CENTER = ROI_W / 2.0                 # where the robot sits in the lane ROI

def synthetic_frame(
        marks,
        mark_width: int = 6,
        road: int = 60,
        surround: int = 30,
        marking: int = 240,
    ) -> np.ndarray:
    """
    BGR frame whose lane ROI holds markings at known ROI-local x, so the correct offset is known exactly.

    Purpose:
        Synthetic ground truth proves the arithmetic recovers what was drawn.
        It says nothing about whether the camera sees the world the way these
        frames assume, so it can't test accuracy in cm against the +/-2 cm
        requirement; that needs captures at measured lateral offsets on a real
        course.

    Inputs:
        marks: ROI-local x positions, or (x, y_top, y_bottom) tuples to
            control vertical extent for dash and partial-visibility cases.
        mark_width: Marking width in px.
        road, surround, marking: Intensities for the road surface inside the
            lane ROI, everything outside it, and the markings.

    Outputs:
        (FRAME_H, FRAME_W, 3) uint8 BGR. A marking drawn at ROI x is recovered
        within half a pixel: marks at 150 and 290 come back as 149.5 and 289.5
        at 480x270 with MEASURED.
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