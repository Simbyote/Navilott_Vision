"""Main pipeline: camera frame in, navigation packet out, through the debug-free stages only.

Purpose:
    What the robot runs. It declares its own flow through the production
    twins (detect_geometry, detect_color, estimate_lane_offset, fuse), so no
    frame pays for the overlays, traces and debug dicts the linkers build.
    phase2_linker and phase3_linker stay as the instrumented debuggers of the
    same flow; test_pipeline holds the two to identical results, so neither
    can drift. Nothing here reads or writes a file per frame: tuning arrives
    once, as a PipelineConfig from src/config.py.

Main package:
    Pipeline: one run's config and stage flow.
        perceive(frame, frame_id, timestamp_ms) -> Phase2Output   (Phases 1-2)

Flow:
    FrameData -> preprocess_frame -> crop_rois -> detect_geometry
              -> detect_color -> estimate_lane_offset -> fuse
              -> package_phase2 -> Phase2Output
    Stage timing is opt-in (Pipeline(timing=True)); off, the flow pays one
    branch per stage and last_timings_ms stays empty.
"""
import time

import numpy as np

from src.capture.camera import FrameData
from src.config import MEASURED, PipelineConfig
from src.perception.color_branch import detect_color
from src.perception.feature_fusion import fuse
from src.perception.geometry import detect_geometry
from src.perception.lane_offset import estimate_lane_offset
from src.perception.phase2_out import Phase2Output, package_phase2
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import crop_rois


# =============================================================================
# Stage timing
# =============================================================================

class _Laps:
    """
    Wall time per stage for one frame, in ms, keyed as run_chain's timings_ms.

    lap(name) records the time since the previous lap (or start()) under name.
    """
    __slots__ = ("times", "_mark")

    def start(self) -> None:
        self.times = {}
        self._mark = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.times[name] = (now - self._mark) * 1000.0
        self._mark = now


# =============================================================================
# Pipeline
# =============================================================================

class Pipeline:
    """
    The robot's per-frame flow. Create once per run; call perceive() on every frame.

    Inputs:
        config: Every stage's tuning. Defaults to MEASURED, the robot's.
        timing: Record each stage's wall time into last_timings_ms. Off by
            default: the loop shouldn't pay for timers it doesn't read.

    Attributes:
        config: As given; read on every frame.
        last_timings_ms: The latest frame's stage times with timing on
            (preprocess, roi, geometry, color, lane_offset, fusion, package);
            {} with timing off. A fresh dict per frame, so a caller may keep it.
    """

    def __init__(self, config: PipelineConfig = MEASURED, timing: bool = False) -> None:
        self.config = config
        self.last_timings_ms: dict = {}
        self._laps = _Laps() if timing else None

    def perceive(
            self,
            frame_bgr: np.ndarray,
            frame_id: int,
            timestamp_ms: int,
        ) -> Phase2Output:
        """
        Run one frame through Phases 1-2.

        Inputs:
            frame_bgr: (H, W, 3) uint8 BGR, as the camera delivered it. Not modified.
            frame_id, timestamp_ms: The stamp capture assigned. Carried to
                every output; never re-derived.

        Outputs:
            Phase2Output: fused detections, the lane offset result and the
            frame stamp. Identical, field by field, to run_chain(...).phase2
            under the same config.

        Raises:
            Whatever the stages raise on malformed input (wrong shape or
            dtype); see preprocess_frame and detect_geometry.
        """
        cfg, laps = self.config, self._laps
        if laps:
            laps.start()

        pre = preprocess_frame(FrameData(frame_bgr, frame_id, timestamp_ms), cfg.preprocess)
        if laps:
            laps.lap("preprocess")

        roi = crop_rois(pre, cfg.roi)
        if laps:
            laps.lap("roi")

        geo = detect_geometry(roi, cfg.geometry)
        if laps:
            laps.lap("geometry")

        traffic = detect_color(roi, cfg.color)
        if laps:
            laps.lap("color")

        offset = estimate_lane_offset(geo, roi, cfg.lane_offset)
        if laps:
            laps.lap("lane_offset")

        fusion = fuse(geo, traffic, roi)
        if laps:
            laps.lap("fusion")

        phase2 = package_phase2(fusion, offset)
        if laps:
            laps.lap("package")
            self.last_timings_ms = laps.times
        return phase2
