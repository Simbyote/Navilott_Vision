"""Main pipeline: camera frame in, navigation packet out, through the debug-free stages only.

Purpose:
    What the robot runs. It declares its own flow through the production
    twins (detect_geometry, detect_color, estimate_lane_offset,
    estimate_stop_line_distance, fuse), so no
    frame pays for the overlays, traces and debug dicts the linkers build.
    phase2_linker and phase3_linker stay as the instrumented debuggers of the
    same flow; test_pipeline holds the two to identical results, so neither
    can drift. Nothing here reads or writes a file per frame: tuning arrives
    once, as a PipelineConfig from src/config.py.

Main package:
    Pipeline: one run's config, Phase 3 state and stage flow.
        perceive(frame, frame_id, timestamp_ms) -> Phase2Output      (Phases 1-2)
        estimate(phase2, sensors)               -> EstimationPacket  (Phase 3)
        step(frame, frame_id, timestamp_ms, sensors) -> EstimationPacket  (both)

Flow:
    FrameData -> preprocess_frame -> crop_rois -> detect_geometry
              -> detect_color -> estimate_lane_offset
              -> estimate_stop_line_distance -> fuse
              -> package_phase2 -> Phase2Output
              -> Phase3Processor.process -> EstimationPacket
    Phase 3 runs as one stage: its internal order belongs to estimation.py.
    Stage timing is opt-in (Pipeline(timing=True)); off, the flow pays one
    branch per stage and last_timings_ms stays empty.
"""
import time

import numpy as np

from dataclasses import replace

from src.capture.camera import FrameData
from src.config import MEASURED, MEASURED_ESTIMATION, PipelineConfig
from src.estimation import EstimationPacket, Phase3Config, Phase3Processor, SensorSample
from src.params import FRAME_H, FRAME_W
from src.perception.color_branch import detect_color
from src.perception.feature_fusion import fuse
from src.perception.geometry import detect_geometry
from src.perception.lane_offset import estimate_lane_offset
from src.perception.phase2_out import Phase2Output, package_phase2
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import crop_rois, resolve
from src.perception.stop_line_distance import estimate_stop_line_distance


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
    The robot's per-frame flow. Create once per run; call step() (or
    perceive() then estimate()) on every frame, in order.

    Inputs:
        config: Phase 2 tuning. Defaults to MEASURED, the robot's.
        estimation: Phase 3 tuning. Defaults to MEASURED_ESTIMATION. When it
            sets cm_per_px but not lane_roi_width_px, the width is taken
            from config's lane ROI at frame_size, and perceive() then
            refuses frames of any other size, since the cm scale would be
            wrong for them.
        timing: Record each stage's wall time into last_timings_ms. Off by
            default: the loop shouldn't pay for timers it doesn't read.
        frame_size: (height, width) the camera delivers. Only used to derive
            the lane ROI width.

    Attributes:
        config: As given; read on every frame.
        estimation: As given, with lane_roi_width_px filled in if derived.
        processor: This run's Phase3Processor. Stateful across frames.
        last_estimation_debug: Phase 3's debug summary for the latest
            estimate() (frame_id, timestamp_ms, dt, log); None before the first.
        last_timings_ms: With timing on, the latest frame's stage times
            (preprocess, roi, geometry, color, lane_offset, stop_line, fusion, package,
            then phase3 once estimated); {} with timing off. A fresh dict
            per frame, so a caller may keep it.
    """

    def __init__(
            self,
            config: PipelineConfig = MEASURED,
            estimation: Phase3Config = MEASURED_ESTIMATION,
            timing: bool = False,
            frame_size: tuple[int, int] = (FRAME_H, FRAME_W),
        ) -> None:
        self.config = config
        self._frame_size = None
        if estimation.cm_per_px is not None and estimation.lane_roi_width_px is None:
            lane_w = resolve(config.roi.lane, frame_size)[2]
            estimation = replace(estimation, lane_roi_width_px=int(lane_w))
            self._frame_size = tuple(frame_size)
        self.estimation = estimation
        self.processor = Phase3Processor(estimation)
        self.last_estimation_debug: dict | None = None
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
            ValueError: If the lane ROI width was derived for the cm scale
                and this frame is another size.
            Whatever the stages raise on malformed input (wrong shape or
            dtype); see preprocess_frame and detect_geometry.
        """
        cfg, laps = self.config, self._laps
        if self._frame_size is not None and frame_bgr.shape[:2] != self._frame_size:
            raise ValueError(f"Pipeline: frame is {frame_bgr.shape[:2]}, but the cm scale "
                             f"was set up for {self._frame_size}")
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

        stop_line = estimate_stop_line_distance(geo, roi, cfg.stop_line, cfg.ground, cfg.stop_line_table)
        if laps:
            laps.lap("stop_line")

        fusion = fuse(geo, traffic, roi)
        if laps:
            laps.lap("fusion")

        phase2 = package_phase2(fusion, offset, stop_line)
        if laps:
            laps.lap("package")
            self.last_timings_ms = laps.times
        return phase2

    def estimate(
            self,
            phase2: Phase2Output,
            sensors: SensorSample | None = None,
        ) -> EstimationPacket:
        """
        Run one frame's Phase 2 output through Phase 3.

        Inputs:
            phase2: This frame's Phase2Output, in frame order: Phase 3
                smooths, holds and votes across calls.
            sensors: Readings for this frame window; None runs without sensors.

        Outputs:
            EstimationPacket for Navigation. Phase 3's debug summary is kept
            on last_estimation_debug; with timing on, its time is added to
            last_timings_ms as "phase3".
        """
        laps = self._laps
        if laps:
            t0 = time.perf_counter()
        packet, self.last_estimation_debug = self.processor.process(phase2, sensors)
        if laps:
            self.last_timings_ms["phase3"] = (time.perf_counter() - t0) * 1000.0
        return packet

    def step(
            self,
            frame_bgr: np.ndarray,
            frame_id: int,
            timestamp_ms: int,
            sensors: SensorSample | None = None,
        ) -> EstimationPacket:
        """
        One frame through Phases 1-3: perceive() then estimate().

        Inputs:
            frame_bgr, frame_id, timestamp_ms: As for perceive().
            sensors: As for estimate().

        Outputs:
            EstimationPacket. Per-stage times, with timing on, are on
            last_timings_ms; Phase 3's debug on last_estimation_debug.
        """
        return self.estimate(self.perceive(frame_bgr, frame_id, timestamp_ms), sensors)
