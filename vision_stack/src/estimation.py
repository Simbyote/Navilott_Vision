"""Phase 3 estimation: per-frame perception plus sensors into a navigation packet.

Purpose:
    Phase 2 answers "what is in this frame"; Phase 3 answers "what is true
    across the last few frames, given the sensors". A stage belongs here only
    if it uses history (smoothing, voting, holding) or sensor data; per-frame
    judgments about pixels belong in Phase 2. Each stage is a small class that
    owns only its own state, so stages test alone and can't reach into each
    other. peripherals.imu is never imported (it loads board drivers at
    import time), so this runs and tests off the Pi.

Main package:
    EstimationPacket: smoothed lane offset and its status, heading change
    since vision was lost, the voted drive state and stop-sign flag, sensor
    pass-throughs, and the frame identity carried from Phase2Output.

Flow (Phase3Processor.process() is the only place the order is written):
    1. LaneFilter: EMA, per-frame jump gate and dropout hold on the lane offset.
    2. HeadingTracker: integrate IMU yaw while vision is lost; zero on vision.
    3. TrafficClassifier: confidence gate and majority vote -> go / caution / stop.
    4. StopSignClassifier: confidence gate and majority vote -> bool.
    5. Assemble the EstimationPacket.
"""
from collections import Counter, deque
from dataclasses import dataclass

from src.params import (
    GREEN, MODE_LEFT_ONLY, MODE_RIGHT_ONLY, MODE_TWO_BOUNDARY, RED, STOP_SIGN,
    TRAFFIC_LIGHT, YELLOW,
)
from src.utils import clamp
from src.perception.feature_fusion import DetectionObject
from src.perception.lane_offset import LaneOffsetResult
from src.perception.phase2_out import Phase2Output

# LaneOffsetResult modes that carry a measurement. "none" and
# "single_uncalibrated" report offset 0.0 with no information behind it
USABLE_LANE_MODES = (MODE_TWO_BOUNDARY, MODE_LEFT_ONLY, MODE_RIGHT_ONLY)

# Lane status reported to Navigation
LANE_VISION = "vision"   # this frame's offset is a fresh measurement
LANE_HOLD = "hold"       # vision dropped; repeating the last good offset
LANE_STALE = "stale"     # held too long; Navigation must not steer by it

# Drive states
GO, CAUTION, STOP = "go", "caution", "stop"
_DRIVE_FOR_COLOR = {RED: STOP, YELLOW: CAUTION, GREEN: GO}


@dataclass(frozen=True)
class Phase3Config:
    """Tuning for Phase 3."""
    ema_alpha: float = 0.35                 # lane smoothing; higher follows faster and smooths less
    max_offset_jump: float | None = 0.5     # largest believable normalized change between frames; a bigger one is a dropout. None disables
    hold_max_frames: int = 7                # frames to repeat the last good offset before it goes stale; at 20 FPS, about 350 ms
    lane_roi_width_px: int | None = None    # undoes the offset normalization; None leaves lane_offset_cm unset
    cm_per_px: float | None = None          # ground scale at the bottom of the lane ROI, hand-measured until calibration exists; None leaves lane_offset_cm unset

    # Frames in the traffic-light and stop-sign votes. A state changes only on
    # a strict majority of the full window, so 3 means 2 of the last 3 frames.
    vote_window: int = 3
    min_confidence_traffic: float = 0.40    # gate on the fused traffic light confidence
    min_confidence_sign: float = 0.45       # gate on the fused stop sign confidence

    gyro_bias_dps: float = 0.0              # gyro Z at standstill, subtracted before integrating
    heading_limit_deg: float = 90.0         # clamp on the integrated heading
    max_dt_s: float = 0.5                   # longer frame gaps are clamped so a stall can't integrate a large heading step


@dataclass(frozen=True)
class SensorSample:
    """Sensor readings for one frame window. None means not available."""
    yaw_rate_dps: float | None = None          # mean gyro Z over the window, deg/s; + = turning right
    lateral_accel_mps2: float | None = None    # signed accel Y with the largest |a| over the window, m/s^2
    wheel_speed_mps: float | None = None       # wheel encoders, m/s; not wired yet, so only carried through

    @classmethod
    def from_imu(
            cls,
            imu_frame,
            wheel_speed_mps: float | None = None,
        ) -> "SensorSample":
        """
        Build a SensorSample from a peripherals.imu.IMUFrame.

        Inputs:
            imu_frame: Duck-typed on valid, mean_yaw_rate_dps and
                peak_lateral_accel, so this module never imports the IMU
                driver. None or an invalid frame (no samples) gives None readings.
            wheel_speed_mps: Carried through as-is.
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
    """Output of the lane filter for one frame."""
    offset: float               # smoothed, normalized; the last good value while holding or stale, 0.0 before any measurement
    offset_cm: float | None     # None unless Phase3Config sets the scale
    status: str                 # LANE_VISION | LANE_HOLD | LANE_STALE

@dataclass(frozen=True)
class EstimationPacket:
    """Phase 3 -> Navigation handoff for one frame. frame_id and timestamp_ms are carried from Phase2Output, never re-derived."""
    # [-1, 1] of half the lane ROI width, + = robot right of lane center.
    # Already a usable steering error. Steer only when lane_status is "vision" or "hold".
    lane_offset: float
    lane_offset_cm: float | None    # same in cm; None until a scale is configured
    lane_status: str                # "vision" | "hold" | "stale"
    heading_error: float            # deg turned since the last frame on vision, from the IMU; 0.0 on vision; + = turned right
    drive_state: str                # "go" | "caution" | "stop", from the traffic light vote
    stop_sign_detected: bool        # from the stop sign vote
    yaw_rate: float                 # pass-through, deg/s; 0.0 if unavailable
    lateral_accel: float            # pass-through, m/s^2; 0.0 if unavailable
    wheel_speed: float              # pass-through, m/s; 0.0 if unavailable
    frame_id: int
    timestamp_ms: int


class _EMA:
    """Exponential moving average of one scalar. value is None until seeded"""
    def __init__(self) -> None:
        self.value: float | None = None

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
    Majority vote with hold.

    The state changes only when one value holds a strict majority of the full
    window; otherwise the previous state stands. Measuring against the window
    rather than the samples seen so far means one frame at startup can't flip
    the state, and a three-way split doesn't pick a winner at random.
    """
    def __init__(self, window: int, initial) -> None:
        self._window = max(1, window)
        self._buf: deque = deque(maxlen=self._window)
        self.state = initial

    def update(self, sample):
        self._buf.append(sample)
        value, count = Counter(self._buf).most_common(1)[0]
        if 2 * count > self._window:
            self.state = value
        return self.state


class LaneFilter:
    """
    Smooths lane offset and bridges short vision dropouts.

    A frame counts as a measurement only if its mode is usable and its offset
    is within max_offset_jump of the current estimate; anything else is a
    dropout. When the hold runs out the EMA is cleared, so the next usable
    frame re-seeds the estimate instead of being rejected by the jump gate forever.
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._ema = _EMA()
        self._last = 0.0
        self._missed = cfg.hold_max_frames + 1   # stale until first measurement

    def _to_cm(self, offset: float) -> float | None:
        """Undo the half-width normalization and scale to cm; None without a scale."""
        cfg = self._cfg
        if cfg.lane_roi_width_px is None or cfg.cm_per_px is None:
            return None
        return round(offset * (cfg.lane_roi_width_px / 2.0) * cfg.cm_per_px, 2)

    def update(
            self,
            results: list[LaneOffsetResult],
            log: list[str],
        ) -> LaneEstimate:
        """
        Fold this frame's lane offset result into the estimate.

        Inputs:
            results: Phase2Output.lane_offset_results (zero or one entry).

        Outputs:
            LaneEstimate for this frame.

        Side effects:
            Appends the reason to log when the frame is a dropout.
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
            # @TODO dead-reckon from wheel odometry during the hold once the encoders are wired
            self._missed += 1
            if self._missed <= cfg.hold_max_frames:
                status = LANE_HOLD
            else:
                status = LANE_STALE
                self._ema.reset()

        offset = round(self._last, 4)
        return LaneEstimate(offset, self._to_cm(offset), status)


class HeadingTracker:
    """
    Integrates gyro yaw while vision is lost.

    While the lane filter is on vision, the offset already carries the
    correction, so the heading reference resets to zero. Once vision drops,
    integrating yaw rate says how far the robot has turned since the last
    good frame, which is what a recovery maneuver needs. This is a change in
    heading, not an absolute heading relative to the lane.
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._heading = 0.0

    def update(
            self,
            lane_status: str,
            yaw_rate_dps: float | None,
            dt: float,
            log: list[str],
        ) -> float:
        """
        Degrees turned since vision was lost, clamped to +/- heading_limit_deg.

        Inputs:
            lane_status: LANE_VISION resets the heading to 0.
            yaw_rate_dps: None holds the heading and logs it.
            dt: Seconds since the previous frame; 0 integrates nothing.
        """
        cfg = self._cfg
        if lane_status == LANE_VISION:
            self._heading = 0.0
        elif yaw_rate_dps is None:
            log.append("[HEADING] no yaw rate this frame; heading held")
        elif dt > 0.0:
            self._heading += (yaw_rate_dps - cfg.gyro_bias_dps) * dt
            limit = cfg.heading_limit_deg
            self._heading = clamp(self._heading, -limit, limit)
        return round(self._heading, 3)


class TrafficClassifier:
    """
    Traffic light detection -> voted drive state.

    Fusion already keeps at most one traffic light per frame, so there is no
    candidate selection here. A frame with no gated light, or an unknown
    color, votes "go".
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, GO)

    def update(
            self,
            detections: list[DetectionObject],
            log: list[str],
        ) -> str:
        """Voted drive state after this frame. Logs lights below the confidence gate."""
        raw = GO
        for d in detections:
            if d.type != TRAFFIC_LIGHT:
                continue
            if d.confidence < self._cfg.min_confidence_traffic:
                log.append(f"[TRAFFIC] {d.label_detail} conf={d.confidence:.3f} "
                           f"below {self._cfg.min_confidence_traffic}")
                continue
            raw = _DRIVE_FOR_COLOR.get(d.label_detail, GO)
        return self._vote.update(raw)


class StopSignClassifier:
    """Stop sign detection -> voted bool."""
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, False)

    def update(
            self,
            detections: list[DetectionObject],
            log: list[str],
        ) -> bool:
        """Voted stop-sign flag after this frame. Logs signs below the confidence gate."""
        raw = False
        for d in detections:
            if d.type != STOP_SIGN:
                continue
            if d.confidence < self._cfg.min_confidence_sign:
                log.append(f"[SIGN] conf={d.confidence:.3f} "
                           f"below {self._cfg.min_confidence_sign}")
                continue
            raw = True
        return self._vote.update(raw)


class Phase3Processor:
    """
    Runs every Phase 3 stage for one frame. Create once; call process() on
    every frame in order.
    """
    def __init__(self, config: Phase3Config | None = None) -> None:
        self._cfg = config or Phase3Config()
        self.lane = LaneFilter(self._cfg)
        self.heading = HeadingTracker(self._cfg)
        self.traffic = TrafficClassifier(self._cfg)
        self.stop_sign = StopSignClassifier(self._cfg)
        self._last_ts: int | None = None

    def _dt(self, timestamp_ms: int, log: list[str]) -> float:
        """Seconds since the previous frame, clamped to [0, max_dt_s]"""
        prev, self._last_ts = self._last_ts, timestamp_ms
        if prev is None:
            return 0.0
        dt = (timestamp_ms - prev) / 1000.0
        if dt < 0.0 or dt > self._cfg.max_dt_s:
            log.append(f"[DT] {dt:.3f}s outside [0, {self._cfg.max_dt_s}]; clamped")
        return clamp(dt, 0.0, self._cfg.max_dt_s)

    def process(
            self,
            phase2: Phase2Output,
            sensors: SensorSample | None = None,
        ) -> tuple[EstimationPacket, dict]:
        """
        Run one Phase 3 cycle.

        Inputs:
            phase2: This frame's Phase2Output.
            sensors: Readings for the same frame window; None means no sensors.

        Outputs:
            (packet, debug_summary). debug_summary holds frame_id,
            timestamp_ms, dt and log.

        Raises:
            ValueError: If phase2 is None.
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