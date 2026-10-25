"""Main pipeline: camera frame in, motor command out, through the debug-free stages only.

Purpose:
    What the robot runs. It declares the flow and nothing else, so no frame
    pays for the linkers' overlays and traces; test_pipeline holds the two to
    identical results.

Main package:
    Pipeline.step(frame, frame_id, timestamp_ms, sensors) -> Command.

Flow:
    1. Phase 2: preprocess, ROIs, the detectors (color and obstacle on a side
       thread when parallel), lane offset, stop line, fusion, package.
    2. Phase 3: Phase3Processor.process -> EstimationPacket.
    3. Navigation.update -> enforce -> Command; the caller drives the motors.
"""
import time

import numpy as np

from src.capture.camera import FrameData
from src.config import MEASURED, MEASURED_ESTIMATION, PipelineConfig
from src.estimation.estimation import EstimationPacket, Phase3Config, Phase3Processor, with_lane_roi_width
from src.navigation.navigation import Command, Navigation, enforce
from src.navigation.route import Route
from src.params import FRAME_H, FRAME_W
from src.perception.parallel import enabled as parallel_enabled, run_side, side_pool
from src.perception.color_branch import detect_color
from src.perception.feature_fusion import fuse
from src.perception.geometry import detect_geometry
from src.perception.lane_offset import estimate_lane_offset
from src.perception.obstacle import estimate_obstacle
from src.perception.phase2_out import Phase2Output, package_phase2
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import crop_rois
from src.perception.stop_line_distance import estimate_stop_line_distance
from src.peripherals.sensing import SensorSample
from src.utils import Laps, check_frame_size


class Pipeline:
    """
    Create once per run; call step() on every frame, in order. When estimation sets
    cm_per_px, the lane ROI width comes from frame_size and other frame sizes are refused.
    timing=True fills last_timings_ms (and last_geometry_ms, last_side_ms) per frame.
    last_packet, last_problems and last_estimation_debug describe the latest frame.
    """

    def __init__(
            self,
            config: PipelineConfig = MEASURED,
            estimation: Phase3Config = MEASURED_ESTIMATION,
            route: Route | None = None,
            timing: bool = False,
            frame_size: tuple[int, int] = (FRAME_H, FRAME_W),
            parallel: bool | None = None,
        ) -> None:
        self.config = config
        self.parallel = parallel_enabled(parallel)
        self.last_side_ms: dict = {}
        self.last_geometry_ms: dict = {}
        self.estimation = with_lane_roi_width(estimation, config.roi, frame_size)
        # A derived width holds every frame to the size it was derived at
        self._frame_size = None if self.estimation is estimation else tuple(frame_size)
        self.processor = Phase3Processor(self.estimation)
        self.navigation = Navigation(gyro_bias_dps=self.estimation.gyro_bias_dps, route=route)
        self.last_packet: EstimationPacket | None = None
        self.last_problems: list[str] = []
        self.last_estimation_debug: dict | None = None
        self.last_timings_ms: dict = {}
        self._laps = Laps() if timing else None

    @property
    def finished(self) -> bool:
        return self.navigation.finished

    def warm_up(self, frame_bgr: np.ndarray) -> float:
        """
        One frame through Phase 2, thrown away, so the first driving frame doesn't pay the
        first-call costs (1.4-1.5 s on 2026-10-20). Phase 3 and navigation aren't touched. Returns ms.
        """
        t0 = time.perf_counter()
        self.perceive(frame_bgr, 0, 0)
        return (time.perf_counter() - t0) * 1000.0

    def perceive(
            self,
            frame_bgr: np.ndarray,
            frame_id: int,
            timestamp_ms: int,
        ) -> Phase2Output:
        """
        The same Phase2Output as run_chain(...).phase2 under the same config. ValueError for a frame of another size.
        """
        cfg, laps = self.config, self._laps
        check_frame_size(frame_bgr, self._frame_size, "Pipeline")
        if laps:
            laps.start()

        pre = preprocess_frame(FrameData(frame_bgr, frame_id, timestamp_ms), cfg.preprocess)
        if laps:
            laps.lap("preprocess")

        roi = crop_rois(pre, cfg.roi)
        if laps:
            laps.lap("roi")

        if self.parallel:
            return self._perceive_parallel(roi, cfg, laps)

        geo = detect_geometry(roi, cfg.geometry, self._geometry_timings(laps))
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

        obstacle = estimate_obstacle(roi, cfg.obstacle, cfg.ground, cfg.stop_line_table)
        if laps:
            laps.lap("obstacle")

        fusion = fuse(geo, traffic, roi)
        if laps:
            laps.lap("fusion")

        phase2 = package_phase2(fusion, offset, stop_line, obstacle)
        if laps:
            laps.lap("package")
            self.last_timings_ms = laps.times
        return phase2

    def _geometry_timings(self, laps) -> dict | None:
        if not laps:
            return None
        self.last_geometry_ms = {}
        return self.last_geometry_ms

    def _perceive_parallel(self, roi, cfg: PipelineConfig, laps) -> Phase2Output:
        side = side_pool().submit(run_side, [
            ("color", lambda: detect_color(roi, cfg.color)),
            ("obstacle", lambda: estimate_obstacle(roi, cfg.obstacle, cfg.ground, cfg.stop_line_table)),
        ])
        geo = detect_geometry(roi, cfg.geometry, self._geometry_timings(laps))
        if laps:
            laps.lap("geometry")
        offset = estimate_lane_offset(geo, roi, cfg.lane_offset)
        if laps:
            laps.lap("lane_offset")
        stop_line = estimate_stop_line_distance(geo, roi, cfg.stop_line, cfg.ground, cfg.stop_line_table)
        if laps:
            laps.lap("stop_line")
        results, self.last_side_ms = side.result()
        if laps:
            laps.lap("branch_wait")
        fusion = fuse(geo, results["color"], roi)
        if laps:
            laps.lap("fusion")
        phase2 = package_phase2(fusion, offset, stop_line, results["obstacle"])
        if laps:
            laps.lap("package")
            self.last_timings_ms = laps.times
        return phase2

    def estimate(
            self,
            phase2: Phase2Output,
            sensors: SensorSample | None = None,
        ) -> EstimationPacket:
        """Frames in order: Phase 3 smooths, holds and votes across calls. sensors=None runs without."""
        laps = self._laps
        if laps:
            laps.mark()
        packet, self.last_estimation_debug = self.processor.process(phase2, sensors)
        if laps:
            laps.lap("phase3")
            self.last_timings_ms = laps.times
        return packet

    def navigate(self, packet: EstimationPacket) -> Command:
        """Navigation's command, or BRAKE if it broke the contract (why: last_problems)."""
        laps = self._laps
        if laps:
            laps.mark()
        cmd, self.last_problems = enforce(self.navigation.update(packet))
        if laps:
            laps.lap("navigation")
            self.last_timings_ms = laps.times
        return cmd

    def step(
            self,
            frame_bgr: np.ndarray,
            frame_id: int,
            timestamp_ms: int,
            sensors: SensorSample | None = None,
        ) -> Command:
        self.last_packet = self.estimate(self.perceive(frame_bgr, frame_id, timestamp_ms), sensors)
        return self.navigate(self.last_packet)
