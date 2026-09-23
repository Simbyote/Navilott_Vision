"""
estimation.py

Phase 3 - Estimation

Purpose:
    Turns one Phase2Output per frame, plus the sensor readings from the same
    frame window, into an EstimationPacket for the Navigation subsystem.

    Phase 2 answers "what is in this frame". Phase 3 answers "what is true
    across the last few frames, given the sensors". A stage belongs here only
    if it uses history (smoothing, voting, holding) or sensor data. Anything
    that is a per-frame judgment about pixels belongs in Phase 2.

Stages:
    1. LaneFilter          lane_offset_results[0] -> smoothed offset
                           EMA, a per-frame jump gate, and a dropout hold
    2. HeadingTracker      IMU yaw integrated while vision is lost. Zero while
                           the lane filter is on vision
    3. TrafficClassifier   traffic_light detection -> "go" | "caution" | "stop"
                           confidence gate + majority vote
    4. StopSignClassifier  stop_sign detection -> bool
                           confidence gate + majority vote
    5. Packet assembly     Phase3Processor.process()

    Each stage is a small class that owns only its own state, so stages can be
    tested alone and cannot reach into each other. Phase3Processor owns one of
    each and is the only place the stage order is written down.

Inputs:
    Phase2Output from perception.phase2_out.package_phase2()
    SensorSample for the same frame window; every field is optional

Units:
    lane_offset stays normalized, [-1, 1] of half the lane ROI width, positive
    when the robot is right of lane center. It is already a usable steering
    error. lane_offset_cm is filled only when Phase3Config has both
    lane_roi_width_px and cm_per_px set, and is None otherwise.

Sensor seams:
    Wheel encoders are not wired yet. wheel_speed_mps is carried through when
    present; nothing depends on it. The dropout hold repeats the last good
    offset rather than dead-reckoning from odometry. That is the place to add
    encoder-based prediction once the data exists.

    This module does not import peripherals.imu, which pulls in the board
    drivers at import time. SensorSample.from_imu() reads an IMUFrame by its
    fields instead, so estimation runs and tests off the Pi.
"""
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from src.perception.feature_fusion import DetectionObject
from src.perception.lane_offset import LaneOffsetResult
from src.perception.phase2_out import Phase2Output

# =============================================================================
# Constants
# =============================================================================
# LaneOffsetResult modes that carry a measurement. "none" and
# "single_uncalibrated" report offset 0.0 with no information behind it
USABLE_LANE_MODES = ("two_boundary", "left_only", "right_only")

# Lane status reported to Navigation
LANE_VISION = "vision"   # this frame's offset is a fresh measurement
LANE_HOLD = "hold"       # vision dropped; repeating the last good offset
LANE_STALE = "stale"     # held too long; Navigation must not steer by it

# Drive states
GO, CAUTION, STOP = "go", "caution", "stop"
_DRIVE_FOR_COLOR = {"red": STOP, "yellow": CAUTION, "green": GO}

# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class Phase3Config:
    """
    Tuning for Phase 3

    ema_alpha: lane offset smoothing. Higher follows faster, smooths less
    max_offset_jump: largest believable change in normalized offset between
                     consecutive frames. A bigger jump is treated as a dropout
                     frame, not a measurement. None disables the gate
    hold_max_frames: frames to repeat the last good offset before reporting
                     it stale. At 20 FPS, 7 frames is about 350 ms
    lane_roi_width_px: lane ROI width, to undo the offset normalization.
                       None leaves lane_offset_cm unset
    cm_per_px: ground scale at the bottom of the lane ROI, where lane offset
               measures. Hand-measured until calibration exists. None leaves
               lane_offset_cm unset
    vote_window: frames in the traffic light and stop sign majority votes.
                 A state changes only when it holds a strict majority of the
                 window, so 3 means 2 of the last 3 frames
    min_confidence_traffic: gate on the fused traffic light confidence
    min_confidence_sign: gate on the fused stop sign confidence
    gyro_bias_dps: gyro Z reading at standstill, subtracted before integrating
    heading_limit_deg: clamp on the integrated heading
    max_dt_s: a frame gap longer than this is clamped, so a stalled frame
              does not integrate a large heading step
    """
    ema_alpha: float = 0.35
    max_offset_jump: Optional[float] = 0.5
    hold_max_frames: int = 7
    lane_roi_width_px: Optional[int] = None
    cm_per_px: Optional[float] = None

    vote_window: int = 3
    min_confidence_traffic: float = 0.40
    min_confidence_sign: float = 0.45

    gyro_bias_dps: float = 0.0
    heading_limit_deg: float = 90.0
    max_dt_s: float = 0.5

# =============================================================================
# Input / Output Dataclasses
# =============================================================================
@dataclass(frozen=True)
class SensorSample:
    """
    Sensor readings for one frame window. None means not available

    yaw_rate_dps: mean gyro Z over the window, deg/s, + = turning right
    lateral_accel_mps2: peak |accel Y| over the window, m/s^2
    wheel_speed_mps: from the wheel encoders, m/s (not wired yet)
    """
    yaw_rate_dps: Optional[float] = None
    lateral_accel_mps2: Optional[float] = None
    wheel_speed_mps: Optional[float] = None

    @classmethod
    def from_imu(
            cls,
            imu_frame,
            wheel_speed_mps: Optional[float] = None,
        ) -> "SensorSample":
        """
        Purpose:
            Build a SensorSample from a peripherals.imu.IMUFrame

        Notes:
            Duck-typed on the IMUFrame fields so this module never imports the
            IMU driver. An invalid frame (no samples) gives None readings
        """
        if imu_frame is None or not imu_frame.valid:
            return cls(wheel_speed_mps=wheel_speed_mps)
        return cls(
            yaw_rate_dps = imu_frame.mean_yaw_rate_dps,
            lateral_accel_mps2 = imu_frame.peak_lateral_accel,
            wheel_speed_mps = wheel_speed_mps,
        )

@dataclass(frozen=True)
class LaneEstimate:
    """
    Output of the lane filter for one frame

    offset: smoothed normalized offset; the last good value while holding or
            stale, 0.0 before any measurement
    offset_cm: offset in cm, None unless Phase3Config sets the scale
    status: LANE_VISION | LANE_HOLD | LANE_STALE
    """
    offset: float
    offset_cm: Optional[float]
    status: str

@dataclass(frozen=True)
class EstimationPacket:
    """
    Phase 3 -> Navigation handoff for one frame

    lane_offset: smoothed normalized offset from lane center, [-1, 1]
                 + = robot right of center. Steer only when lane_status is
                 "vision" or "hold"
    lane_offset_cm: same in cm, None until a scale is configured
    lane_status: "vision" | "hold" | "stale"
    heading_error: deg turned since the last frame on vision, from the IMU.
                   0.0 while on vision. + = turned right
    drive_state: "go" | "caution" | "stop", from the traffic light vote
    stop_sign_detected: from the stop sign vote
    yaw_rate: pass-through, deg/s, 0.0 if unavailable
    lateral_accel: pass-through, m/s^2, 0.0 if unavailable
    wheel_speed: pass-through, m/s, 0.0 if unavailable
    frame_id: carried from Phase2Output, never re-derived
    timestamp_ms: carried from Phase2Output, never re-derived
    """
    lane_offset: float
    lane_offset_cm: Optional[float]
    lane_status: str
    heading_error: float
    drive_state: str
    stop_sign_detected: bool
    yaw_rate: float
    lateral_accel: float
    wheel_speed: float
    frame_id: int
    timestamp_ms: int

# =============================================================================
# Shared Filters
# =============================================================================
class _EMA:
    """Exponential moving average of one scalar. value is None until seeded"""
    def __init__(self) -> None:
        self.value: Optional[float] = None

    def update(self, sample: float, alpha: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = alpha * sample + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None

class _Vote:
    """
    Majority vote with hold

    The state changes only when one value holds a strict majority of the full
    window. Otherwise the previous state stands. Measuring against the window
    rather than the samples seen so far means one frame at startup cannot flip
    the state, and a three-way split does not pick a winner at random
    """
    def __init__(self, window: int, initial) -> None:
        self._window = max(1, window)
        self._buf: Deque = deque(maxlen=self._window)
        self.state = initial

    def update(self, sample):
        self._buf.append(sample)
        value, count = Counter(self._buf).most_common(1)[0]
        if 2 * count > self._window:
            self.state = value
        return self.state

# =============================================================================
# Stage 1: Lane Filter
# =============================================================================
class LaneFilter:
    """
    Smooths lane offset and bridges short vision dropouts

    Notes:
        A frame counts as a measurement only if its mode is usable and its
        offset is within max_offset_jump of the current estimate. Anything
        else is a dropout frame. When the hold runs out the EMA is cleared,
        so the next usable frame re-seeds the estimate instead of being
        rejected by the jump gate forever
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._ema = _EMA()
        self._last = 0.0
        self._missed = cfg.hold_max_frames + 1   # stale until first measurement

    def _to_cm(self, offset: float) -> Optional[float]:
        cfg = self._cfg
        if cfg.lane_roi_width_px is None or cfg.cm_per_px is None:
            return None
        return round(offset * (cfg.lane_roi_width_px / 2.0) * cfg.cm_per_px, 2)

    def update(
            self,
            results: List[LaneOffsetResult],
            log: list,
        ) -> LaneEstimate:
        """
        Purpose:
            Fold this frame's lane offset result into the estimate

        Inputs:
            results: Phase2Output.lane_offset_results (zero or one entry)
            log: debug log, appended to

        Outputs:
            LaneEstimate
        """
        cfg = self._cfg
        result = results[0] if results else None
        accepted = False

        if result is None:
            log.append("[LANE] no lane offset result supplied")
        elif result.mode not in USABLE_LANE_MODES:
            log.append(f"[LANE] mode {result.mode} carries no measurement")
        elif (cfg.max_offset_jump is not None and self._ema.value is not None
                and abs(result.offset - self._ema.value) > cfg.max_offset_jump):
            log.append(f"[LANE] jump {result.offset - self._ema.value:+.3f} "
                       f"exceeds {cfg.max_offset_jump}; treated as dropout")
        else:
            accepted = True

        if accepted:
            self._last = self._ema.update(result.offset, cfg.ema_alpha)
            self._missed = 0
            status = LANE_VISION
        else:
            self._missed += 1
            if self._missed <= cfg.hold_max_frames:
                status = LANE_HOLD
            else:
                status = LANE_STALE
                self._ema.reset()

        offset = round(self._last, 4)
        return LaneEstimate(offset, self._to_cm(offset), status)

# =============================================================================
# Stage 2: Heading Tracker
# =============================================================================
class HeadingTracker:
    """
    Integrates gyro yaw while vision is lost

    Notes:
        While the lane filter is on vision, the offset already carries the
        correction, so the heading reference resets to zero. Once vision
        drops, integrating yaw rate says how far the robot has turned since
        the last good frame, which is what a recovery maneuver needs. This is
        a change in heading, not an absolute heading relative to the lane
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._heading = 0.0

    def update(
            self,
            lane_status: str,
            yaw_rate_dps: Optional[float],
            dt: float,
            log: list,
        ) -> float:
        cfg = self._cfg
        if lane_status == LANE_VISION:
            self._heading = 0.0
        elif yaw_rate_dps is None:
            log.append("[HEADING] no yaw rate this frame; heading held")
        elif dt > 0.0:
            self._heading += (yaw_rate_dps - cfg.gyro_bias_dps) * dt
            limit = cfg.heading_limit_deg
            self._heading = max(-limit, min(limit, self._heading))
        return round(self._heading, 3)

# =============================================================================
# Stage 3: Traffic Classifier
# =============================================================================
class TrafficClassifier:
    """
    Traffic light detection -> voted drive state

    Notes:
        Fusion already keeps at most one traffic light per frame, so there is
        no candidate selection here. A frame with no gated light votes "go"
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, GO)

    def update(
            self,
            detections: List[DetectionObject],
            log: list,
        ) -> str:
        raw = GO
        for d in detections:
            if d.type != "traffic_light":
                continue
            if d.confidence < self._cfg.min_confidence_traffic:
                log.append(f"[TRAFFIC] {d.label_detail} conf={d.confidence:.3f} "
                           f"below {self._cfg.min_confidence_traffic}")
                continue
            raw = _DRIVE_FOR_COLOR.get(d.label_detail, GO)
        return self._vote.update(raw)

# =============================================================================
# Stage 4: Stop Sign Classifier
# =============================================================================
class StopSignClassifier:
    """Stop sign detection -> voted bool"""
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, False)

    def update(
            self,
            detections: List[DetectionObject],
            log: list,
        ) -> bool:
        raw = False
        for d in detections:
            if d.type != "stop_sign":
                continue
            if d.confidence < self._cfg.min_confidence_sign:
                log.append(f"[SIGN] conf={d.confidence:.3f} "
                           f"below {self._cfg.min_confidence_sign}")
                continue
            raw = True
        return self._vote.update(raw)

# =============================================================================
# Phase 3 Processor
# =============================================================================
class Phase3Processor:
    """
    Runs every Phase 3 stage for one frame. Create once; call process() on
    every frame in order
    """
    def __init__(self, config: Optional[Phase3Config] = None) -> None:
        self._cfg = config or Phase3Config()
        self.lane = LaneFilter(self._cfg)
        self.heading = HeadingTracker(self._cfg)
        self.traffic = TrafficClassifier(self._cfg)
        self.stop_sign = StopSignClassifier(self._cfg)
        self._last_ts: Optional[int] = None

    def _dt(self, timestamp_ms: int, log: list) -> float:
        """Seconds since the previous frame, clamped to [0, max_dt_s]"""
        prev, self._last_ts = self._last_ts, timestamp_ms
        if prev is None:
            return 0.0
        dt = (timestamp_ms - prev) / 1000.0
        if dt < 0.0 or dt > self._cfg.max_dt_s:
            log.append(f"[DT] {dt:.3f}s outside [0, {self._cfg.max_dt_s}]; clamped")
        return max(0.0, min(dt, self._cfg.max_dt_s))

    def process(
            self,
            phase2: Phase2Output,
            sensors: Optional[SensorSample] = None,
        ) -> tuple:
        """
        Purpose:
            Run one Phase 3 cycle

        Inputs:
            phase2: Phase2Output for this frame
            sensors: SensorSample for the same window; None means no sensors

        Outputs:
            packet: EstimationPacket
            debug_summary: dict --- "frame_id", "timestamp_ms", "dt", "log"
        """
        if phase2 is None:
            raise ValueError("Phase3Processor.process: phase2 output is None")
        sensors = sensors or SensorSample()
        log = []

        dt = self._dt(phase2.timestamp_ms, log)
        lane = self.lane.update(phase2.lane_offset_results, log)
        heading = self.heading.update(lane.status, sensors.yaw_rate_dps, dt, log)
        drive_state = self.traffic.update(phase2.detections, log)
        stop_sign = self.stop_sign.update(phase2.detections, log)

        packet = EstimationPacket(
            lane_offset = lane.offset,
            lane_offset_cm = lane.offset_cm,
            lane_status = lane.status,
            heading_error = heading,
            drive_state = drive_state,
            stop_sign_detected = stop_sign,
            yaw_rate = sensors.yaw_rate_dps or 0.0,
            lateral_accel = sensors.lateral_accel_mps2 or 0.0,
            wheel_speed = sensors.wheel_speed_mps or 0.0,
            frame_id = phase2.frame_id,
            timestamp_ms = phase2.timestamp_ms,
        )
        return packet, {
            "frame_id": phase2.frame_id,
            "timestamp_ms": phase2.timestamp_ms,
            "dt": round(dt, 4),
            "log": log,
        }