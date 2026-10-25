"""Main pipeline: camera frame in, motor command out, through the debug-free stages only.

Purpose:
    What the robot runs. It declares the flow and nothing else: each stage
    is a call into the file that owns it, so no frame pays for the overlays,
    traces and debug dicts the linkers build. The linkers stay as the
    instrumented twins of the same flow, and test_pipeline holds them to
    identical results. Nothing here touches a file per frame.

Main package:
    Pipeline: one run's config, Phase 3 and Navigation state;
    step(frame, frame_id, timestamp_ms, sensors) -> Command, or perceive /
    estimate / navigate one phase at a time.

Flow:
    1. Phase 2: preprocess, ROIs, geometry, color, lane offset, stop line,
       obstacle, fusion, package (parallel: color and obstacle on a side thread).
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
    The robot's per-frame flow. Create once per run; call step() (or
    perceive(), estimate() then navigate()) on every frame, in order.

    Inputs:
        config: Phase 2 tuning. Defaults to MEASURED, the robot's.
        estimation: Phase 3 tuning. Defaults to MEASURED_ESTIMATION. When it
            sets cm_per_px but not lane_roi_width_px, the width is taken
            from config's lane ROI at frame_size (with_lane_roi_width), and
            perceive() then refuses frames of any other size, since the cm
            scale would be wrong for them.
        route: The course plan for Navigation; None is no maneuvers,
            finishing at the mat's edge. Navigation's heading hold takes
            its gyro bias from estimation.gyro_bias_dps, as Phase 3 does.
        timing: Record each stage's wall time into last_timings_ms. Off by
            default: the loop shouldn't pay for timers it doesn't read.
        frame_size: (height, width) the camera delivers. Only used to derive
            the lane ROI width.
        parallel: Color and obstacle on Phase 2's side thread while geometry
            runs here (perception/parallel.py); None is the run's choice, else
            params.PARALLEL_PERCEPTION.

    Attributes:
        config: As given; read on every frame.
        estimation: As given, with lane_roi_width_px filled in if derived.
        processor: This run's Phase3Processor. Stateful across frames.
        navigation: This run's Navigation. Stateful across frames; its
            outcome, end_step and progress say how the run went.
        last_packet: The latest step()'s EstimationPacket; None before the first.
        last_problems: How the latest navigate()'s command broke the
            contract ([] almost always); it was braked instead.
        last_estimation_debug: Phase 3's debug summary for the latest
            estimate() (frame_id, timestamp_ms, dt, log); None before the first.
        last_timings_ms: With timing on, the latest frame's stage times
            (preprocess, roi, geometry, color, lane_offset, stop_line, obstacle, fusion, package,
            then phase3 and navigation once run); {} with timing off. A fresh
            dict per frame, so a caller may keep it. Parallel, the main
            thread's only (color and obstacle replaced by branch_wait, so
            they still add up to the frame), and last_side_ms has the side
            thread's own color and obstacle times.
        last_geometry_ms: With timing on, the latest frame's geometry split
            into its parts (geometry.GEOMETRY_PARTS: the shared edges, then
            each detector), inside last_timings_ms["geometry"]; {} with timing off.
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
        """Navigation has ended the run (navigation.outcome says how); every command since is BRAKE."""
        return self.navigation.finished

    def warm_up(self, frame_bgr: np.ndarray) -> float:
        """
        Run one frame through Phases 1-2 and throw the result away, so the
        first driving frame doesn't pay the stages' first-call costs (the
        2026-10-20 runs' frame 0 took 1.4-1.5 s, obstacle alone 1,089 ms).
        Phase 2 keeps nothing between frames; Phase 3 and navigation aren't
        touched, so the run starts exactly as without it.

        Outputs:
            How long it took, ms.
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
        Run one frame through Phases 1-2.

        Inputs:
            frame_bgr: (H, W, 3) uint8 BGR, as the camera delivered it. Not modified.
            frame_id, timestamp_ms: The stamp capture assigned. Carried to
                every output; never re-derived.

        Outputs:
            Phase2Output: fused detections, the lane offset, stop-line and
            obstacle results, and the frame stamp. Identical, field by field, to run_chain(...).phase2
            under the same config.

        Raises:
            ValueError: If the lane ROI width was derived for the cm scale
                and this frame is another size.
            Whatever the stages raise on malformed input (wrong shape or
            dtype); see preprocess_frame and detect_geometry.
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
        """A fresh dict for detect_geometry to time its parts into, kept as last_geometry_ms; None untimed."""
        if not laps:
            return None
        self.last_geometry_ms = {}
        return self.last_geometry_ms

    def _perceive_parallel(self, roi, cfg: PipelineConfig, laps) -> Phase2Output:
        """perceive() from the ROIs on: color and obstacle on the side thread, the rest here."""
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
            laps.mark()
        packet, self.last_estimation_debug = self.processor.process(phase2, sensors)
        if laps:
            laps.lap("phase3")
            self.last_timings_ms = laps.times
        return packet

    def navigate(self, packet: EstimationPacket) -> Command:
        """
        Run one frame's packet through Navigation.

        Inputs:
            packet: This frame's EstimationPacket, in frame order: the
                rules, the route and the end of course keep state across calls.

        Outputs:
            The Command to drive: Navigation's, or BRAKE if it broke the
            contract (the reasons on last_problems). With timing on, its
            time is added to last_timings_ms as "navigation".
        """
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
        """
        One frame through the whole chain: perceive(), estimate(), navigate().

        Inputs:
            frame_bgr, frame_id, timestamp_ms: As for perceive().
            sensors: As for estimate().

        Outputs:
            The Command to drive. The packet it came from is on
            last_packet; per-stage times, with timing on, on
            last_timings_ms; Phase 3's debug on last_estimation_debug.
        """
        self.last_packet = self.estimate(self.perceive(frame_bgr, frame_id, timestamp_ms), sensors)
        return self.navigate(self.last_packet)
