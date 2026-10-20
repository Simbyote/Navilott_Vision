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
    since vision was lost, the voted drive state, stop-sign flag,
    stop-line flag with its distance, obstacle flag with its distance,
    sensor pass-throughs (IMU yaw rate and
    lateral accel, each wheel's encoder counts per second), and the frame
    identity carried from Phase2Output.
    with_lane_roi_width(): the lane ROI width the cm scale needs, from Phase
    2's ROI tuning; the pipeline and phase3_linker both build Phase 3 with it.

Flow (Phase3Processor.process() is the only place the order is written):
    1. LaneFilter: EMA, per-frame jump gate and dropout hold on the lane offset.
    2. HeadingTracker: integrate IMU yaw while vision is lost; zero on vision.
    3. TrafficClassifier: confidence gate and majority vote -> go / caution / stop.
    4. StopSignClassifier: confidence gate and majority vote -> bool.
    5. StopLineClassifier: majority vote -> bool, with the distance held
       through frames that miss the line while the vote stands.
    6. ObstacleClassifier: the same for the obstacle in the robot's path.
    7. Odometer: encoder counts -> distance driven, cm (odometry.py).
    8. Assemble the EstimationPacket.
"""
from collections import Counter, deque
from dataclasses import dataclass, replace

from src.params import (
    GREEN, MODE_LEFT_ONLY, MODE_NONE, MODE_RIGHT_ONLY, MODE_TWO_BOUNDARY, RED, STOP_SIGN,
    TRAFFIC_LIGHT, YELLOW,
)
from src.utils import clamp
from src.estimation.odometry import Odometer
from src.perception.feature_fusion import DetectionObject
from src.perception.lane_offset import LaneOffsetResult
from src.perception.obstacle import ObstacleResult
from src.perception.phase2_out import Phase2Output
from src.perception.roi_crop import ROIConfig, resolve
from src.perception.stop_line_distance import StopLineResult

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

    cm_per_count: float | None = None       # floor cm per encoder count (calibration/odometry.json); None leaves every distance unset


def with_lane_roi_width(estimation: Phase3Config, roi: ROIConfig,
                        frame_size: tuple[int, int]) -> Phase3Config:
    """
    Fill in lane_roi_width_px for the cm scale.

    Inputs:
        estimation: The Phase 3 tuning.
        roi: Phase 2's ROI tuning; the lane ROI's width is read from it.
        frame_size: (height, width) of the frames Phase 2 will crop.
    Outputs:
        estimation with lane_roi_width_px set to the lane ROI's width at
        frame_size when it sets cm_per_px and leaves the width unset;
        estimation itself (the same object) otherwise.
    """
    if estimation.cm_per_px is None or estimation.lane_roi_width_px is not None:
        return estimation
    return replace(estimation, lane_roi_width_px=int(resolve(roi.lane, frame_size)[2]))


@dataclass(frozen=True)
class SensorSample:
    """Sensor readings for one frame window. None means not available."""
    yaw_rate_dps: float | None = None          # mean gyro Z over the window, deg/s; + = turning right
    lateral_accel_mps2: float | None = None    # signed accel Y with the largest |a| over the window, m/s^2
    left_wheel_cps: float | None = None        # left encoder counts/s over the frame window; + = forward
    right_wheel_cps: float | None = None       # right encoder counts/s over the frame window; + = forward
    left_count: int | None = None              # cumulative left count at the window's end; + = forward
    right_count: int | None = None             # cumulative right count at the window's end
    # Degrees turned over the window, integrated reading by reading (raw: the
    # gyro bias is subtracted over yaw_span_s). None leaves the heading to
    # yaw_rate_dps x the frame step, as replays without it do
    yaw_deg: float | None = None
    yaw_span_s: float | None = None

    @classmethod
    def from_batch(cls, batch) -> "SensorSample":
        """
        Build a SensorSample from one frame window's sensing.SensorBatch.

        Inputs:
            batch: Duck-typed on mean_yaw_dps (already + = turning right),
                peak_lateral_accel, left_cps, right_cps, left_count,
                right_count, yaw_deg and yaw_span_s, so this module never
                imports sensing. Each is None when its sensor isn't there or
                gave nothing this window.
        """
        return cls(
            yaw_rate_dps = batch.mean_yaw_dps,
            lateral_accel_mps2 = batch.peak_lateral_accel,
            left_wheel_cps = batch.left_cps,
            right_wheel_cps = batch.right_cps,
            left_count = getattr(batch, "left_count", None),
            right_count = getattr(batch, "right_count", None),
            yaw_deg = getattr(batch, "yaw_deg", None),
            yaw_span_s = getattr(batch, "yaw_span_s", None),
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
    stop_line_detected: bool        # from the stop line vote
    # Lane-ROI rows from the nearest stop line to the ROI bottom (0 = on it);
    # None unless stop_line_detected. Held from the last frame that saw it
    stop_line_distance_px: float | None
    # Floor cm forward to where the line crosses the robot's centerline, held
    # like the px value; None also when there is no ground homography
    stop_line_distance_cm: float | None
    yaw_rate: float                 # pass-through, deg/s; 0.0 if unavailable
    lateral_accel: float            # pass-through, m/s^2; 0.0 if unavailable
    frame_id: int
    timestamp_ms: int
    # Encoder counts per second over the frame window, + = forward; 0.0 when
    # stopped or with no encoders (detecting the encoders is the drivers' job)
    left_wheel_cps: float
    right_wheel_cps: float
    # Pass-through: this frame's Phase 2 lane offset mode (params MODE_*), before
    # any filtering; MODE_TWO_BOUNDARY means both lane boundaries were seen.
    # Navigation's intersection crossing ends on it
    lane_mode: str = MODE_NONE
    # From the obstacle vote (perception/obstacle.py): something solid in the
    # robot's path. Its distances are held like the stop line's: rows from its
    # base to the lane ROI bottom, and floor cm (None without a ground
    # homography or stop-line table); clipped: its base is at or past the
    # view bottom, so it is as near as the view shows or nearer. All four
    # are False / None whenever the vote says there's none
    obstacle_detected: bool = False
    obstacle_distance_px: float | None = None
    obstacle_distance_cm: float | None = None
    obstacle_clipped: bool = False
    # Odometry (odometry.py): floor cm driven since the run's first encoder
    # reading, + = forward; the axle's midpoint, and each wheel. None
    # without encoders or without Phase3Config.cm_per_count
    distance_cm: float | None = None
    left_distance_cm: float | None = None
    right_distance_cm: float | None = None
    # The voted stop line's tilt in the image, this frame's measurement only
    # (never held), deg from horizontal, + = its right end nearer the robot.
    # None unless stop_line_detected and seen this frame. The intersection
    # rule reads the approach angle from it
    stop_line_tilt_deg: float | None = None
    # Pass-through: degrees turned over this frame's sensor window, integrated
    # reading by reading, raw (the gyro bias is the reader's, over yaw_span_s).
    # None without it; readers then fall back to yaw_rate x the frame step
    yaw_deg: float | None = None
    yaw_span_s: float | None = None


def lane_mode_of(phase2: Phase2Output) -> str:
    """The frame's raw Phase 2 lane mode; MODE_NONE when Phase 2 gave no lane offset result."""
    results = phase2.lane_offset_results
    return results[0].mode if results else MODE_NONE


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

    @property
    def buffer(self) -> tuple:
        """The samples in the window, oldest first. Read-only; for the debug twin."""
        return tuple(self._buf)

    @property
    def window(self) -> int:
        return self._window

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

    def _jumps(self, offset: float) -> bool:
        """True if offset is further than max_offset_jump from the estimate (never while it's unseeded)."""
        cfg = self._cfg
        return (cfg.max_offset_jump is not None and self._ema.value is not None
                and abs(offset - self._ema.value) > cfg.max_offset_jump)

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
        offset = None if result is None else result.offset

        if result is None:
            log.append("[LANE] no lane offset result supplied")
        elif result.mode not in USABLE_LANE_MODES:
            log.append(f"[LANE] mode {result.mode} carries no measurement")
        elif self._jumps(offset):
            # One boundary's side is a guess from where it sits in the ROI;
            # past the lane's edge, its right line is left of center. The other
            # side, if it fits the estimate, is the line it is.
            other = getattr(result, "other_side_offset", None)
            if other is not None and not self._jumps(other):
                log.append(f"[LANE] {result.mode} {offset:+.3f} breaks continuity; "
                           f"the line taken as the other side: {other:+.3f}")
                offset, accepted = other, True
            else:
                log.append(f"[LANE] jump {offset - self._ema.value:+.3f} "
                           f"exceeds {cfg.max_offset_jump}; treated as dropout")
        else:
            accepted = True

        if accepted:
            self._last = self._ema.update(offset, cfg.ema_alpha)
            self._missed = 0
            status = LANE_VISION
        else:
            # @TODO dead-reckon from wheel odometry during the hold (the encoders' counts reach the sensor sample)
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
            yaw_deg: float | None = None,
            yaw_span_s: float | None = None,
        ) -> float:
        """
        Degrees turned since vision was lost, clamped to +/- heading_limit_deg.

        Inputs:
            lane_status: LANE_VISION resets the heading to 0.
            yaw_rate_dps: None (and no yaw_deg) holds the heading and logs it.
            dt: Seconds since the previous frame, already clamped to
                max_dt_s; 0 integrates nothing. Used only without yaw_deg.
            yaw_deg, yaw_span_s: The window's turn, integrated reading by
                reading, and the seconds it covers. When given, the heading
                steps by exactly that (less the bias over the span), however
                long the frame gap was: a stall loses only readings the hub
                couldn't keep.
        """
        cfg = self._cfg
        if lane_status == LANE_VISION:
            self._heading = 0.0
            return 0.0
        if yaw_deg is not None:
            step = yaw_deg - cfg.gyro_bias_dps * (yaw_span_s or 0.0)
        elif yaw_rate_dps is None:
            log.append("[HEADING] no yaw rate this frame; heading held")
            step = 0.0
        else:
            step = (yaw_rate_dps - cfg.gyro_bias_dps) * dt
        if step:
            limit = cfg.heading_limit_deg
            self._heading = clamp(self._heading + step, -limit, limit)
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


class StopLineClassifier:
    """
    Stop-line results -> voted bool and the distances to report.

    The confidence gate is Phase 2's (StopLineDistanceConfig), since the
    measurement already refuses weak candidates. While the vote says a line
    is there, a frame that misses it repeats the last measured distances (px
    and cm together, from the same frame); both are None whenever the vote
    says there isn't one. tilt is this frame's measured tilt only: never
    held, None on a missed frame or while the vote says there's no line.
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, False)
        self._last: tuple[float | None, float | None] = (None, None)
        self.tilt: float | None = None

    def update(
            self,
            results: list[StopLineResult],
            log: list[str],
        ) -> tuple[bool, float | None, float | None]:
        """(voted flag, distance_px, distance_cm) after this frame. Logs a held distance."""
        seen = next((r for r in results if r.detected), None)
        if seen is not None:
            self._last = (seen.distance_px, seen.distance_cm)
        voted = self._vote.update(seen is not None)
        self.tilt = seen.tilt_deg if voted and seen is not None else None
        if not voted:
            return False, None, None
        if seen is None:
            log.append(f"[STOPLINE] missed this frame; holding {self._last[0]:.1f}px")
        return True, *self._last


class ObstacleClassifier:
    """
    Obstacle results -> voted bool and the measurement to report.

    As StopLineClassifier: Phase 2's gates decide what counts, the vote
    decides whether it's there, and while it is, a frame that misses it
    repeats the last measurement (px, cm and clipped, from one frame). All
    are None / False whenever the vote says there's none.
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, False)
        self._last: tuple[float | None, float | None, bool] = (None, None, False)

    def update(
            self,
            results: list[ObstacleResult],
            log: list[str],
        ) -> tuple[bool, float | None, float | None, bool]:
        """(voted flag, distance_px, distance_cm, clipped) after this frame. Logs a held measurement."""
        seen = next((r for r in results if r.detected), None)
        if seen is not None:
            self._last = (seen.distance_px, seen.distance_cm, seen.clipped)
        if not self._vote.update(seen is not None):
            return False, None, None, False
        if seen is None:
            log.append(f"[OBSTACLE] missed this frame; holding {self._last[0]:.1f}px")
        return True, *self._last


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
        self.stop_line = StopLineClassifier(self._cfg)
        self.obstacle = ObstacleClassifier(self._cfg)
        self.odometry = Odometer(self._cfg.cm_per_count)
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
        heading = self.heading.update(lane.status, sensors.yaw_rate_dps, dt, log, sensors.yaw_deg, sensors.yaw_span_s)
        drive_state = self.traffic.update(phase2.detections, log)
        stop_sign = self.stop_sign.update(phase2.detections, log)
        stop_line, stop_line_px, stop_line_cm = self.stop_line.update(phase2.stop_line_results, log)
        obstacle, obstacle_px, obstacle_cm, obstacle_clipped = self.obstacle.update(phase2.obstacle_results, log)
        odom = self.odometry.update(sensors.left_count, sensors.right_count, log)

        packet = EstimationPacket(
            lane_offset = lane.offset,
            lane_offset_cm = lane.offset_cm,
            lane_status = lane.status,
            heading_error = heading,
            drive_state = drive_state,
            stop_sign_detected = stop_sign,
            stop_line_detected = stop_line,
            stop_line_distance_px = stop_line_px,
            stop_line_distance_cm = stop_line_cm,
            yaw_rate = sensors.yaw_rate_dps or 0.0,
            lateral_accel = sensors.lateral_accel_mps2 or 0.0,
            frame_id = phase2.frame_id,
            timestamp_ms = phase2.timestamp_ms,
            left_wheel_cps = sensors.left_wheel_cps or 0.0,
            right_wheel_cps = sensors.right_wheel_cps or 0.0,
            lane_mode = lane_mode_of(phase2),
            obstacle_detected = obstacle,
            obstacle_distance_px = obstacle_px,
            obstacle_distance_cm = obstacle_cm,
            obstacle_clipped = obstacle_clipped,
            distance_cm = odom.distance_cm,
            left_distance_cm = odom.left_cm,
            right_distance_cm = odom.right_cm,
            stop_line_tilt_deg = self.stop_line.tilt,
            yaw_deg = sensors.yaw_deg,
            yaw_span_s = sensors.yaw_span_s,
        )
        return packet, {
            "frame_id": phase2.frame_id,
            "timestamp_ms": phase2.timestamp_ms,
            "dt": round(dt, 4),
            "log": log,
        }