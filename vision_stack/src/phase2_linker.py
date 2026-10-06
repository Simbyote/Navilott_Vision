"""Phase 1-2 linker: the one place the stage order is written down.

Purpose:
    Runs a frame through every Phase 1-2 stage in order, so the live view and
    the tests all go through one chain and can't drift from it. The color
    branch stays off until calibrated HSV ranges are supplied; lane
    boundaries and stop signs flow either way. Every stage's tuning comes in
    as one PipelineConfig from src/config.py, so re-tuning after a camera
    change is a config swap and a re-run, not a code edit.

Main package:
    ChainResult: every stage's output and debug data for one frame, ending in
    the Phase2Output handed to Phase 3, plus per-stage timings.

Flow:
    FrameData -> preprocess_frame -> crop_rois -> run_geometry_stage
              -> run_color_stage -> compute_lane_offset
              -> compute_stop_line_distance -> fuse_detections -> package_phase2

Command line (options from live_view.cli):
    python3 phase2_linker.py --video clip.mp4 --no-display
    python3 phase2_linker.py --frames frames/Sample1
"""
import time

import numpy as np
from dataclasses import dataclass, field, replace

from src.capture.camera import FrameData
from src.perception.preprocess import preprocess_frame, PreprocessResult
from src.perception.roi_crop import crop_rois, ROICropResult
from src.perception.geometry import run_geometry_stage, GeometryBranchResult
from src.perception.color_branch import run_color_stage, load_hsv_ranges
from src.perception.lane_offset import compute_lane_offset, LaneOffsetResult
from src.perception.feature_fusion import fuse_detections, FusionResult
from src.perception.phase2_out import package_phase2, Phase2Output
from src.perception.stop_line_distance import compute_stop_line_distance, StopLineResult
import src.debugger.live_view as live_view
from src.config import MEASURED, PipelineConfig      # re-exported: older imports read them from here


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
    timings_ms: dict = field(default_factory=dict)      # wall ms per stage: preprocess, roi, geometry, color, lane_offset, stop_line, fusion, package
    traffic: list = field(default_factory=list)         # TrafficLightCandidates; [] when the color branch is off
    traffic_debug: dict = field(default_factory=dict)   # masks, counts and trace; just {"enabled": False} when off
    stop_line: StopLineResult | None = None             # nearest stop line; its candidates are in geometry, their debug in lane_debug["stop_line"]
    stop_line_debug: dict = field(default_factory=dict) # compute_stop_line_distance's debug summary


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

    stop_line, stop_line_debug = compute_stop_line_distance(geo, roi, config.stop_line, config.ground,
                                                            config.stop_line_table)
    lap("stop_line")

    fusion, fusion_debug = fuse_detections(geo, traffic, roi)
    lap("fusion")

    phase2 = package_phase2(fusion, offset, stop_line)
    lap("package")

    return ChainResult(fd, pre, roi, geo, offset, lane_debug, offset_debug,
                       sign_debug, fusion, fusion_debug, phase2, timings,
                       traffic, traffic_debug, stop_line, stop_line_debug)


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
        hsv_path: HSV ranges JSON that replaces config.color's ranges for
            this run; None keeps whatever config.color says (MEASURED
            already loads calibration/hsv_ranges.json).
        options: out_dir, display, scale, stride, limit, fps, views; passed
            to live_view.run().

    Outputs:
        live_view.RunStats.

    Side effects:
        Reads hsv_path. Whatever live_view.run() does with the options:
        display windows and recordings under out_dir.
    """
    if hsv_path:
        config = replace(config, color=replace(config.color, hsv_ranges=load_hsv_ranges(hsv_path)))

    def process(frame_bgr, frame_id, timestamp_ms):
        return run_chain(frame_bgr, frame_id, timestamp_ms, config, trace=trace)

    return live_view.run(source, process, config.lane_offset, **options)


if __name__ == "__main__":
    import sys
    sys.exit(live_view.cli(run_live_view))