"""Phase 1-2 linker: the one place the stage order is written down, with every debug dict.

Purpose:
    The live view, the other linkers and the tests all run frames through
    this one chain, so none can drift from it; pipeline.py is its debug-free
    twin. On the command line it's the live view
    (python3 -m src.phase2_linker, phase2_linker.md).

Main package:
    ChainResult: every stage's output and debug for one frame, ending in the Phase2Output.
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
from src.perception.obstacle import compute_obstacle, ObstacleResult
from src.perception.parallel import enabled as parallel_enabled, run_side, side_pool
import src.debugger.live_view as live_view
from src.config import MEASURED, PipelineConfig      # re-exported: older imports read them from here


@dataclass(frozen=True)
class ChainResult:
    frame: FrameData
    pre: PreprocessResult
    roi: ROICropResult
    geometry: GeometryBranchResult
    offset: LaneOffsetResult
    lane_debug: dict
    offset_debug: dict
    sign_debug: dict = field(default_factory=dict)
    fusion: FusionResult | None = None
    fusion_debug: dict = field(default_factory=dict)
    phase2: Phase2Output | None = None
    timings_ms: dict = field(default_factory=dict)      # wall ms per stage
    traffic: list = field(default_factory=list)         # [] when the color branch is off
    traffic_debug: dict = field(default_factory=dict)
    stop_line: StopLineResult | None = None             # its candidates' debug is in lane_debug["stop_line"]
    stop_line_debug: dict = field(default_factory=dict)
    obstacle: ObstacleResult | None = None
    obstacle_debug: dict = field(default_factory=dict)
    side_timings_ms: dict = field(default_factory=dict) # parallel only
    geometry_timings_ms: dict = field(default_factory=dict)  # parts, inside timings_ms["geometry"]


def run_chain(
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: int = 0,
        config: PipelineConfig = PipelineConfig(),
        draw_overlays: bool = False,
        trace: bool = False,
        parallel: bool | None = None,
    ) -> ChainResult:
    """
    config defaults to PipelineConfig(); run_live_view() uses MEASURED. Results are the same
    parallel or not; parallel, timings_ms has branch_wait in place of color and obstacle, whose
    own times are in side_timings_ms, so timings_ms still adds up to the frame.
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

    side_ms, geo_ms = {}, {}
    if parallel_enabled(parallel):
        side = side_pool().submit(run_side, [
            ("color", lambda: run_color_stage(roi, config.color, trace)),
            ("obstacle", lambda: compute_obstacle(roi, config.obstacle, config.ground, config.stop_line_table)),
        ])
        geo, lane_debug, sign_debug = run_geometry_stage(roi, config.geometry, draw_overlays, trace, geo_ms)
        lap("geometry")
        offset, offset_debug = compute_lane_offset(geo, roi, config.lane_offset)
        lap("lane_offset")
        stop_line, stop_line_debug = compute_stop_line_distance(geo, roi, config.stop_line, config.ground,
                                                                config.stop_line_table)
        lap("stop_line")
        results, side_ms = side.result()
        lap("branch_wait")
        (traffic, traffic_debug), (obstacle, obstacle_debug) = results["color"], results["obstacle"]
    else:
        geo, lane_debug, sign_debug = run_geometry_stage(
            roi, config.geometry, draw_overlays, trace, geo_ms
        )
        lap("geometry")

        traffic, traffic_debug = run_color_stage(roi, config.color, trace)
        lap("color")

        offset, offset_debug = compute_lane_offset(geo, roi, config.lane_offset)
        lap("lane_offset")

        stop_line, stop_line_debug = compute_stop_line_distance(geo, roi, config.stop_line, config.ground,
                                                                config.stop_line_table)
        lap("stop_line")

        obstacle, obstacle_debug = compute_obstacle(roi, config.obstacle, config.ground, config.stop_line_table)
        lap("obstacle")

    fusion, fusion_debug = fuse_detections(geo, traffic, roi)
    lap("fusion")

    phase2 = package_phase2(fusion, offset, stop_line, obstacle)
    lap("package")

    return ChainResult(fd, pre, roi, geo, offset, lane_debug, offset_debug,
                       sign_debug, fusion, fusion_debug, phase2, timings,
                       traffic, traffic_debug, stop_line, stop_line_debug,
                       obstacle, obstacle_debug, side_ms, geo_ms)


def warm_up(frame_bgr: np.ndarray, config: PipelineConfig = PipelineConfig()) -> float:
    """
    One frame through the chain, thrown away, so the first driving frame doesn't pay the
    first-call costs (1.4-1.5 s on 2026-10-20). Phase 2 keeps nothing between frames. Returns ms.
    """
    t0 = time.perf_counter()
    run_chain(frame_bgr, 0, 0, config)
    return (time.perf_counter() - t0) * 1000.0


def run_live_view(source, config: PipelineConfig = MEASURED, trace: bool = True,
                  hsv_path: str | None = None, **options) -> "live_view.RunStats":
    """
    run_chain() handed to live_view.run() as a callable, so live_view never imports this
    module. hsv_path replaces config.color's ranges for the run. Returns live_view.RunStats.
    """
    if hsv_path:
        config = replace(config, color=replace(config.color, hsv_ranges=load_hsv_ranges(hsv_path)))

    def process(frame_bgr, frame_id, timestamp_ms):
        return run_chain(frame_bgr, frame_id, timestamp_ms, config, trace=trace)

    return live_view.run(source, process, config.lane_offset, **options)


if __name__ == "__main__":
    import sys
    sys.exit(live_view.cli(run_live_view))