"""
estimation.py  (was phase3.py)

Navigation Signal Processing

Purpose:
    Receives a Phase2Snapshot from the vision pipeline and produces an
    EstimationPacket for the Navigation subsystem.

Pipeline stages
  1. Motion consistency check  - reject a lane estimate that moved faster than
                                 physically plausible; reject traffic/sign
                                 detections whose centroid jumped too far
  2. Temporal filtering        - EMA smoothing of lane_offset and heading_error;
                                 majority vote on discrete states
  3. Confidence thresholding   - reject any detection below the configured floor
  4. State estimation          - fuse vision with inertial readings and publish
                                 validity metadata alongside every signal

CHANGES FROM THE PREVIOUS REVISION

  A. Phase 3 no longer derives lane_offset. It used to average position_x over
     every surviving lane_boundary detection. Its own docstring said position_x
     was "the signed pixel offset of the LANE CENTRE"; Phase 2 supplied one
     detection PER BOUNDARY. The mean of individual boundary offsets equals the
     lane centre only when the survivors are laterally symmetric, which they
     never are - a dashed centre line fragments into several contours while a
     solid edge line yields one. Reduction now happens once, in
     lane_offset.compute_lane_offset(), and arrives pre-reduced as a
     LaneEstimate.

  B. _CentroidTracker no longer poisons its own reference. It used to write
     last_x/last_y BEFORE the threshold test, so a rejected outlier became the
     next frame's comparand. It also held one slot per detection CLASS while
     being called once per DETECTION - with two lane boundaries present it
     compared the right boundary against the left boundary within the same
     frame, 240 px apart against an 80 px threshold, and rejected both. In
     session2.log, 523 frames (81% of all dead-reckoning holds) had Phase 2
     reporting a lane that Phase 3 silently discarded this way.
     The tracker is retained for traffic_light and stop_sign, where fusion
     already reduces to a single best candidate per class, so one slot per
     class is correct. The lane path uses _LaneRateGate instead.

  C. Validity is now published. lane_confident was computed, used internally to
     pick a heading source, and then dropped. EstimationPacket had no field
     distinguishing a fresh fix from a 39-frame-old one, so Navigation could not
     honour the "must not rely on this value" comment even in principle. The
     packet now carries lane_offset_valid, lane_offset_age, lane_offset_stale
     and lane_mode.

  D. Dead-reckoning now actually expires. Both terminal branches of _filter_lane
     used to return the identical value, making deadreck_max_frames inert.

  E. The EMA is reset after a long gap, so the first frame back from an
     intersection is adopted at full weight instead of being blended 65% with a
     pre-intersection value. That blend is what produced the monotonic ramp to
     +0.2485 (71% of full scale) over the five frames after the f=2268 gap.

  F. Dimensional error fixed in the jump threshold. speed_scale was
     1.0 + wheel_speed * dt * px_per_meter, i.e. a dimensionless literal added
     to a quantity in pixels. Dormant only because wheel_speed is currently
     None; it would have activated silently the moment encoders were wired in.

  G. heading_source is published. heading_error is NOT an independent
     measurement when it is vision-derived - it is lane_offset * HEADING_SCALE.
     Navigation must not fuse the two as two sensors.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from contracts import (
    EstimationPacket, LaneEstimate, NavDetection, Phase2Snapshot, SensorSample,
    LANE_MODE_NONE, LANE_MODES_TWO_ANCHOR,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Phase3Config:
    """
    All tuning parameters for Phase 3.

    1) Temporal filtering
        ema_alpha: EMA factor for lane_offset and heading_error.
                   Higher = faster response, less smoothing. Range [0.1, 0.9].
        vote_window: frames over which discrete states are majority-voted.

    2) Confidence thresholding
        min_confidence_lane / _traffic / _sign

    3) Lane offset scaling
        px_per_meter: SCALING constant only. There is no homography, so this
                   does not survive a change in camera pitch or mount height.
                   Do not treat the output as true metres at range.
        lane_half_width_px: px corresponding to offset_norm = 1.0. Used to put
                   the rate gate in pixel units.

    4) Motion consistency
        max_centroid_jump_px: max centroid displacement between consecutive
                   frames, for traffic_light and stop_sign.
        max_lane_rate_norm_per_s: max plausible rate of change of the
                   normalised lane offset. TODO-CALIBRATE. At 15 FPS a value of
                   6.0 permits ~0.40 of full scale per frame, which is well
                   above real robot dynamics but below a detection swap.

    5) Dead-reckoning
        deadreck_max_frames: frames the last offset may be held before it is
                   flagged stale.
        ema_reset_after_frames: gap length after which the EMA is reset so the
                   reacquisition frame is not blended with a pre-gap value.
        stale_decay_per_frame: fraction of the held offset bled off each frame
                   once stale. 0.0 = pure hold (previous behaviour). 0.10-0.20
                   is recommended once R1-R4 are validated on the bench, so a
                   stale steering bias decays instead of persisting.
    """
    ema_alpha:                float = 0.35
    vote_window:              int   = 3

    min_confidence_lane:      float = 0.30
    min_confidence_traffic:   float = 0.40
    min_confidence_sign:      float = 0.45

    px_per_meter:             float = 686.0
    lane_half_width_px:       float = 240.0

    max_centroid_jump_px:     float = 80.0
    max_lane_rate_norm_per_s: float = 6.0

    deadreck_max_frames:      int   = 10
    ema_reset_after_frames:   int   = 4
    stale_decay_per_frame:    float = 0.0


# ============================================================================
# Internal Filter State
# ============================================================================

@dataclass
class _EMAState:
    """
    Exponential moving average for a single scalar.

    value: current EMA value; None if uninitialised
    """
    value: Optional[float] = None

    def update(self, sample: float, alpha: float) -> float:
        """
        Purpose:
            Update the EMA with a new sample.
        """
        if self.value is None:
            self.value = sample
        else:
            self.value = alpha * sample + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        """
        Purpose:
            Drop history so the next sample is adopted at full weight.
            Called after a vision gap long enough that the retained value is no
            longer evidence about the present.
        """
        self.value = None


@dataclass
class _CentroidTracker:
    """
    Tracks the last accepted centroid for ONE detection class.

    Correct only for classes that fusion reduces to a single best candidate per
    frame: traffic_light and stop_sign. Do not use it for lane boundaries.

    last_x / last_y: last ACCEPTED position; None until the first acceptance
    """
    last_x: Optional[float] = None
    last_y: Optional[float] = None

    def update(self, x: float, y: float, threshold: float) -> bool:
        """
        Purpose:
            Accept a centroid if it is within threshold of the last ACCEPTED
            position.

        Output:
            True if accepted. State is updated only on acceptance, so a
            rejected outlier cannot become the next frame's reference.
        """
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = x, y
            return True

        dx = x - self.last_x
        dy = y - self.last_y
        if (dx * dx + dy * dy) ** 0.5 <= threshold:
            self.last_x, self.last_y = x, y
            return True
        return False

    def reset(self) -> None:
        """
        Purpose:
            Forget the reference. Called after a gap so that reacquisition is
            never judged against a position from before the gap.
        """
        self.last_x = self.last_y = None


@dataclass
class _LaneRateGate:
    """
    Plausibility gate for the pre-reduced lane offset.

    The lane signal is a single scalar per frame, so the right check is a rate
    limit, not a centroid distance. A jump larger than the robot can physically
    produce in one frame means the estimator changed its mind about which
    features are the lane, not that the robot moved.

    last_norm: last ACCEPTED normalised offset; None until first acceptance
    """
    last_norm: Optional[float] = None

    def update(self, offset_norm: float, dt: float, max_rate_norm_per_s: float) -> bool:
        """
        Purpose:
            Accept an offset if it is reachable from the last accepted one.

        Inputs:
            offset_norm: candidate normalised offset
            dt: seconds since the previous frame; <= 0 disables the gate
            max_rate_norm_per_s: plausibility limit

        Output:
            True if accepted; state updated only on acceptance.
        """
        if self.last_norm is None or dt <= 0.0:
            self.last_norm = offset_norm
            return True

        if abs(offset_norm - self.last_norm) <= max_rate_norm_per_s * dt:
            self.last_norm = offset_norm
            return True
        return False

    def reset(self) -> None:
        """
        Purpose:
            Forget the reference after a gap.
        """
        self.last_norm = None


# ============================================================================
# Phase3Processor
# ============================================================================

class Phase3Processor:
    """
    Instantiate once at pipeline startup; call process() on every frame.
    @TODO: Threading support if processing time becomes an issue later on
    """

    def __init__(self, config: Optional[Phase3Config] = None):
        self._cfg = config or Phase3Config()

        self._lane_offset_ema = _EMAState()
        self._heading_error_ema = _EMAState()

        self._drive_state_buf: Deque[str] = deque(maxlen=self._cfg.vote_window)
        self._stop_sign_buf: Deque[bool] = deque(maxlen=self._cfg.vote_window)

        self._lane_gate = _LaneRateGate()
        self._traffic_tracker = _CentroidTracker()
        self._sign_tracker = _CentroidTracker()

        self._last_timestamp_ms: int = 0
        self._lane_age: int = 0                 # frames since last confident fix
        self._last_lane_offset_norm: float = 0.0
        self._last_lane_mode: str = LANE_MODE_NONE

        self._heading_from_imu: float = 0.0

    # =======================================================================
    # Entry point
    # =======================================================================
    def process(
        self,
        phase2_out: Phase2Snapshot,
        sensor_sample: SensorSample,
    ) -> EstimationPacket:
        """
        Purpose:
            Run one Phase 3 cycle.

        Input:
            phase2_out: Phase2Snapshot from the vision pipeline
            sensor_sample: SensorSample for this frame window

        Output:
            EstimationPacket ready for Navigation
        """
        cfg = self._cfg

        # dt for integration and rate limiting
        dt = 0.0
        if self._last_timestamp_ms > 0 and phase2_out.timestamp_ms > 0:
            dt = (phase2_out.timestamp_ms - self._last_timestamp_ms) / 1000.0
            dt = max(0.0, min(dt, 0.5))   # clamp: ignore stale/jumped timestamps
        self._last_timestamp_ms = phase2_out.timestamp_ms

        # Stage 1: motion consistency for the discrete classes
        consistent = self._motion_consistency(
            phase2_out.detections, self._jump_threshold(sensor_sample, dt))

        # Stages 2 & 3: lane
        lane = self._filter_lane(phase2_out.lane, dt)

        # Stages 2 & 3: heading
        heading_error_deg, heading_source = self._filter_heading(
            dt, sensor_sample, lane["valid"])

        # Stages 2 & 3: discrete states
        self._drive_state_buf.append(self._classify_traffic(consistent))
        self._stop_sign_buf.append(self._classify_stop_sign(consistent))

        drive_state = _majority_vote_str(self._drive_state_buf, default="go")
        stop_sign_detected = _majority_vote_bool(self._stop_sign_buf)

        return EstimationPacket(
            lane_offset=round(lane["norm"] * cfg.lane_half_width_px / cfg.px_per_meter, 4),
            lane_offset_norm=round(lane["norm"], 4),
            lane_offset_valid=lane["valid"],
            lane_offset_age=lane["age"],
            lane_offset_stale=lane["stale"],
            lane_mode=lane["mode"],

            heading_error=round(heading_error_deg, 3),
            heading_source=heading_source,

            drive_state=drive_state,
            stop_sign_detected=stop_sign_detected,

            yaw_rate=sensor_sample.yaw_rate or 0.0,
            lateral_accel=sensor_sample.lateral_accel or 0.0,
            wheel_speed=sensor_sample.wheel_speed or 0.0,

            frame_id=phase2_out.frame_id,
            timestamp_ms=phase2_out.timestamp_ms,
        )

    # =======================================================================
    # Stage 1: Motion consistency
    # =======================================================================
    def _jump_threshold(self, sensor_sample: SensorSample, dt: float) -> float:
        """
        Purpose:
            Scale the centroid jump threshold by how far the robot actually
            travelled this frame.

        Notes:
            The previous form was
                1.0 + wheel_speed * dt * px_per_meter
            which added a dimensionless 1.0 to a quantity in pixels and
            saturated its own clamp at any realistic speed. The travelled
            distance in pixels is now divided by the base threshold, making the
            scale factor dimensionless as intended.
        """
        cfg = self._cfg
        if sensor_sample.wheel_speed is None or dt <= 0.0:
            return cfg.max_centroid_jump_px

        travelled_px = sensor_sample.wheel_speed * dt * cfg.px_per_meter
        scale = 1.0 + travelled_px / max(cfg.max_centroid_jump_px, 1.0)
        return cfg.max_centroid_jump_px * min(scale, 3.0)

    def _motion_consistency(
        self,
        detections: List[NavDetection],
        jump_thresh: float,
    ) -> List[NavDetection]:
        """
        Purpose:
            Discard traffic_light / stop_sign detections whose centroid jumped
            further than jump_thresh from the last ACCEPTED position for that
            class.

        Notes:
            Lane boundaries are deliberately absent here. They are reduced to a
            single LaneEstimate before the seam and gated by _LaneRateGate.
        """
        result = []
        for det in detections:
            if det.type == "traffic_light":
                ok = self._traffic_tracker.update(
                    det.position_x, det.position_y, jump_thresh)
            elif det.type == "stop_sign":
                ok = self._sign_tracker.update(
                    det.position_x, det.position_y, jump_thresh)
            else:
                ok = True
            if ok:
                result.append(det)
        return result

    # =======================================================================
    # Stages 2 & 3: Lane offset
    # =======================================================================
    def _filter_lane(self, lane: LaneEstimate, dt: float) -> dict:
        """
        Purpose:
            Smooth and validate the pre-reduced lane estimate.

        Inputs:
            lane: LaneEstimate produced by lane_offset.compute_lane_offset()
            dt: seconds since the previous frame

        Output:
            dict with keys: norm, valid, age, stale, mode
                norm  : smoothed normalised offset, canonical sign convention
                valid : True only if this frame contributed a fresh measurement
                age   : frames since the last fresh measurement (0 = this frame)
                stale : True once age exceeds deadreck_max_frames
                mode  : the lane mode behind this value

        Notes:
            No averaging happens here. Phase 3 consumes the reduction, it does
            not perform it. This is fix A.
        """
        cfg = self._cfg

        usable = (
            lane is not None
            and lane.valid
            and lane.mode != LANE_MODE_NONE
            and lane.confidence >= cfg.min_confidence_lane
        )

        if usable and not self._lane_gate.update(
                lane.offset_norm, dt, cfg.max_lane_rate_norm_per_s):
            usable = False   # implausible jump: treat as a missed frame

        if usable:
            # Long gap behind us: drop EMA history so the reacquisition frame
            # is adopted at full weight rather than blended with a stale value.
            if self._lane_age >= cfg.ema_reset_after_frames:
                self._lane_offset_ema.reset()
                self._heading_error_ema.reset()

            smoothed = self._lane_offset_ema.update(lane.offset_norm, cfg.ema_alpha)
            self._last_lane_offset_norm = smoothed
            self._last_lane_mode = lane.mode
            self._lane_age = 0
            return {"norm": smoothed, "valid": True, "age": 0,
                    "stale": False, "mode": lane.mode}

        # No usable measurement this frame
        self._lane_age += 1
        stale = self._lane_age > cfg.deadreck_max_frames

        if stale and cfg.stale_decay_per_frame > 0.0:
            # Bleed the held offset toward centre so a stale steering bias does
            # not persist indefinitely.
            self._last_lane_offset_norm *= (1.0 - cfg.stale_decay_per_frame)

        # Reset the gates so reacquisition is judged fresh, not against a
        # pre-gap reference.
        if self._lane_age >= cfg.ema_reset_after_frames:
            self._lane_gate.reset()
            self._traffic_tracker.reset()
            self._sign_tracker.reset()

        return {"norm": self._last_lane_offset_norm, "valid": False,
                "age": self._lane_age, "stale": stale,
                "mode": self._last_lane_mode}

    # =======================================================================
    # Stages 2 & 3: Heading error
    # =======================================================================
    def _filter_heading(
        self,
        dt: float,
        sensor_sample: SensorSample,
        lane_confident: bool,
    ) -> tuple:
        """
        Purpose:
            Produce a heading error and say where it came from.

        Output:
            (heading_error_deg, source) where source is "vision" | "imu" | "hold"

        Notes:
            When source == "vision", heading_error is lane_offset * HEADING_SCALE
            and carries NO information independent of lane_offset. It is a
            convenience restatement, not a second sensor. pipeline.md lists its
            primary source as "Vision (lane geometry)"; no lane geometry - slope,
            line angle, vanishing point - is computed anywhere. Contour
            orientation is available from cv2.minAreaRect in geometry.py and is
            currently discarded. Deriving a real heading from it is the correct
            fix and is deliberately out of scope for this change.
        """
        HEADING_SCALE = 30.0   # deg per unit of normalised lateral offset

        if lane_confident and self._lane_offset_ema.value is not None:
            raw_heading = self._lane_offset_ema.value * HEADING_SCALE
            smoothed = self._heading_error_ema.update(raw_heading, self._cfg.ema_alpha)
            self._heading_from_imu = smoothed   # sync accumulator to vision
            return smoothed, "vision"

        if sensor_sample.yaw_rate is not None and dt > 0.0:
            self._heading_from_imu += sensor_sample.yaw_rate * dt
            self._heading_from_imu = max(-90.0, min(90.0, self._heading_from_imu))
            return self._heading_from_imu, "imu"

        held = self._heading_error_ema.value
        return (held if held is not None else 0.0), "hold"

    # =======================================================================
    # Stages 2 & 3: Traffic light state
    # =======================================================================
    def _classify_traffic(self, detections: List[NavDetection]) -> str:
        """
        Purpose:
            Classify traffic light state.

        Output:
            "go" | "caution" | "stop"
        """
        cfg = self._cfg
        tl_dets = [
            d for d in detections
            if d.type == "traffic_light" and d.confidence >= cfg.min_confidence_traffic
        ]
        if not tl_dets:
            return "go"

        best = max(tl_dets, key=lambda d: d.confidence)
        if best.label == "red":
            return "stop"
        if best.label == "yellow":
            return "caution"
        return "go"

    # =======================================================================
    # Stages 2 & 3: Stop sign detection
    # =======================================================================
    def _classify_stop_sign(self, detections: List[NavDetection]) -> bool:
        """
        Purpose:
            Detect stop sign presence above the confidence floor.
        """
        cfg = self._cfg
        return any(
            d.type == "stop_sign" and d.confidence >= cfg.min_confidence_sign
            for d in detections
        )


# ============================================================================
# Helper functions for majority voting
# ============================================================================

def _majority_vote_str(buf: Deque[str], default: str) -> str:
    """
    Purpose:
        Return the most common string in buf; on a tie, the one appearing first.
    """
    if not buf:
        return default
    items = list(buf)
    counts: dict = {}
    for v in items:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: (counts[k], -items.index(k)))


def _majority_vote_bool(buf: Deque[bool]) -> bool:
    """
    Purpose:
        Return True if a strict majority of values in buf are True.
    """
    if not buf:
        return False
    return sum(1 for v in buf if v) > len(buf) / 2