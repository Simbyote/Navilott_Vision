"""Main pipeline: camera frame in, motor command out, through the debug-free stages only.

Purpose:
    What the robot runs. It declares the flow and nothing else: each stage
    is a call into the file that owns it (the production twins of Phase 2,
    Phase 3's processor, the navigation subsystem and the command contract),
    so no frame pays for the overlays, traces and debug dicts the linkers
    build. phase2_linker, phase3_linker and navigation_linker stay as the
    instrumented debuggers of the same flow; test_pipeline holds them to
    identical results, so neither side can drift. Nothing here reads or
    writes a file per frame: tuning arrives once, as a PipelineConfig and
    Phase3Config from src/config.py, and the route as a Route.

Main package:
    Pipeline: one run's config, Phase 3 and Navigation state, and stage flow.
        perceive(frame, frame_id, timestamp_ms) -> Phase2Output      (Phases 1-2)
        estimate(phase2, sensors)               -> EstimationPacket  (Phase 3)
        navigate(packet)                        -> Command           (Navigation)
        step(frame, frame_id, timestamp_ms, sensors) -> Command      (all three)
        finished: Navigation has ended the run.

Flow:
    FrameData -> preprocess_frame -> crop_rois -> detect_geometry
              -> detect_color -> estimate_lane_offset
              -> estimate_stop_line_distance -> fuse
              -> package_phase2 -> Phase2Output
              -> Phase3Processor.process -> EstimationPacket
              -> Navigation.update -> enforce -> Command
    Phase 3 and Navigation each run as one stage: their internal order
    belongs to estimation.py and navigation.py. Driving the motors with the
    Command is the caller's (the run loop's) last stage.
    Stage timing is opt-in (Pipeline(timing=True)); off, the flow pays one
    branch per stage and last_timings_ms stays empty.
"""
import numpy as np

from src.capture.camera import FrameData
from src.config import MEASURED, MEASURED_ESTIMATION, PipelineConfig
from src.estimation.estimation import (
    EstimationPacket, Phase3Config, Phase3Processor, SensorSample, with_lane_roi_width,
)
from src.navigation.navigation import Command, Navigation, enforce
from src.navigation.route import Route
from src.params import FRAME_H, FRAME_W
from src.perception.color_branch import detect_color
from src.perception.feature_fusion import fuse
from src.perception.geometry import detect_geometry
from src.perception.lane_offset import estimate_lane_offset
from src.perception.phase2_out import Phase2Output, package_phase2
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import crop_rois
from src.perception.stop_line_distance import estimate_stop_line_distance
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
            (preprocess, roi, geometry, color, lane_offset, stop_line, fusion, package,
            then phase3 and navigation once run); {} with timing off. A fresh
            dict per frame, so a caller may keep it.
    """

    def __init__(
            self,
            config: PipelineConfig = MEASURED,
            estimation: Phase3Config = MEASURED_ESTIMATION,
            route: Route | None = None,
            timing: bool = False,
            frame_size: tuple[int, int] = (FRAME_H, FRAME_W),
        ) -> None:
        self.config = config
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
        check_frame_size(frame_bgr, self._frame_size, "Pipeline")
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
