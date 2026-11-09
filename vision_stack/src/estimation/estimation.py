"""Phase 3 estimation: per-frame perception plus sensors into a navigation packet.

Purpose:
    Phase 2 says what is in this frame; Phase 3 says what is true across the
    last few frames, given the sensors. A stage belongs here only if it uses
    history or sensor data, and no driver is imported, so it runs off the Pi.

Main package:
    EstimationPacket (field by field: phase3_estimation.md, "Contract").

Flow (Phase3Processor.process() is the only place the order is written):
    1. The lane filter, then the heading while the lane is lost.
    2. The votes: traffic light, stop sign, stop line, obstacle.
    3. Odometry, then the packet.
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
from src.peripherals.sensing import SensorSample

# The modes that carry a measurement
USABLE_LANE_MODES = (MODE_TWO_BOUNDARY, MODE_LEFT_ONLY, MODE_RIGHT_ONLY)

LANE_VISION = "vision"   # this frame's offset is a fresh measurement
LANE_HOLD = "hold"       # vision dropped; repeating the last good offset
LANE_STALE = "stale"     # held too long to steer by

GO, CAUTION, STOP = "go", "caution", "stop"
_DRIVE_FOR_COLOR = {RED: STOP, YELLOW: CAUTION, GREEN: GO}


@dataclass(frozen=True)
class Phase3Config:
    ema_alpha: float = 0.35                 # higher follows faster, smooths less
    max_offset_jump: float | None = 0.5     # a bigger change between frames is a dropout
    hold_max_frames: int = 7                # ~350 ms before the lane goes stale
    lane_roi_width_px: int | None = None    # None leaves lane_offset_cm unset
    cm_per_px: float | None = None          # at the lane ROI's bottom; None leaves lane_offset_cm unset

    # 3: two of the last three frames
    vote_window: int = 3
    min_confidence_traffic: float = 0.40
    min_confidence_sign: float = 0.45

    gyro_bias_dps: float = 0.0              # gyro Z at rest
    heading_limit_deg: float = 90.0
    max_dt_s: float = 0.5                   # so a stall can't integrate a large heading step

    cm_per_count: float | None = None       # None leaves every distance unset


def with_lane_roi_width(estimation: Phase3Config, roi: ROIConfig,
                        frame_size: tuple[int, int]) -> Phase3Config:
    """
    Sets lane_roi_width_px from the ROI when cm_per_px is set and the width isn't; else returns estimation itself.
    """
    if estimation.cm_per_px is None or estimation.lane_roi_width_px is not None:
        return estimation
    return replace(estimation, lane_roi_width_px=int(resolve(roi.lane, frame_size)[2]))


@dataclass(frozen=True)
class LaneEstimate:
    offset: float               # the last good value while holding or stale
    offset_cm: float | None
    status: str


@dataclass(frozen=True)
class EstimationPacket:
    # [-1, 1] of half the lane ROI width, + = robot right of center; steer only on vision or hold
    lane_offset: float
    lane_offset_cm: float | None
    lane_status: str
    heading_error: float            # deg turned since the last vision frame, + = right
    drive_state: str
    stop_sign_detected: bool
    stop_line_detected: bool
    # Held through a missed frame; None unless stop_line_detected
    stop_line_distance_px: float | None
    stop_line_distance_cm: float | None
    yaw_rate: float                 # raw, deg/s; 0.0 if unavailable
    lateral_accel: float            # m/s^2; 0.0 if unavailable
    frame_id: int
    timestamp_ms: int
    # Counts/s over the frame window; 0.0 when stopped or without encoders
    left_wheel_cps: float
    right_wheel_cps: float
    # This frame's raw Phase 2 mode; the intersection crossing ends on it
    lane_mode: str = MODE_NONE
    # Held like the stop line's; clipped: its base is at the view bottom or nearer
    obstacle_detected: bool = False
    obstacle_distance_px: float | None = None
    obstacle_distance_cm: float | None = None
    obstacle_clipped: bool = False
    # Since the run's first encoder reading; None without encoders or a scale
    distance_cm: float | None = None
    left_distance_cm: float | None = None
    right_distance_cm: float | None = None
    # This frame's only, never held; + = right end nearer the robot
    stop_line_tilt_deg: float | None = None
    # Raw degrees over the sensor window; None falls back to yaw_rate x the frame step
    yaw_deg: float | None = None
    yaw_span_s: float | None = None


def lane_mode_of(phase2: Phase2Output) -> str:
    """MODE_NONE when Phase 2 gave no lane result."""
    results = phase2.lane_offset_results
    return results[0].mode if results else MODE_NONE


class _EMA:
    """value is None until seeded."""
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
    Changes state only on a strict majority of the full window, so one frame at startup
    can't flip it and a three-way split picks nothing.
    """
    def __init__(self, window: int, initial) -> None:
        self._window = max(1, window)
        self._buf: deque = deque(maxlen=self._window)
        self.state = initial

    @property
    def buffer(self) -> tuple:
        """Oldest first; for the debug twin."""
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
    A frame is a measurement only if its mode is usable and its offset within
    max_offset_jump of the estimate. When the hold runs out the EMA clears, so the next
    usable frame re-seeds it instead of being refused by the jump gate forever.
    """
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._ema = _EMA()
        self._last = 0.0
        self._missed = cfg.hold_max_frames + 1   # stale until first measurement

    def _to_cm(self, offset: float) -> float | None:
        cfg = self._cfg
        if cfg.lane_roi_width_px is None or cfg.cm_per_px is None:
            return None
        return round(offset * (cfg.lane_roi_width_px / 2.0) * cfg.cm_per_px, 2)

    def _jumps(self, offset: float) -> bool:
        """Never while unseeded."""
        cfg = self._cfg
        return (cfg.max_offset_jump is not None and self._ema.value is not None
                and abs(offset - self._ema.value) > cfg.max_offset_jump)

    def update(
            self,
            results: list[LaneOffsetResult],
            log: list[str],
        ) -> LaneEstimate:
        """Logs the reason for a dropout."""
        cfg = self._cfg
        result = results[0] if results else None
        accepted = False
        offset = None if result is None else result.offset

        if result is None:
            log.append("[LANE] no lane offset result supplied")
        elif result.mode not in USABLE_LANE_MODES:
            log.append(f"[LANE] mode {result.mode} carries no measurement")
        elif self._jumps(offset):
            # One boundary's side is a guess; if the other side fits the estimate, it's that one
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
            self._missed += 1
            if self._missed <= cfg.hold_max_frames:
                status = LANE_HOLD
            else:
                status = LANE_STALE
                self._ema.reset()

        offset = round(self._last, 4)
        return LaneEstimate(offset, self._to_cm(offset), status)


class HeadingTracker:
    """Degrees turned since vision was lost: a change in heading, reset to 0 on every vision frame."""
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
        Clamped to heading_limit_deg. With yaw_deg it steps by exactly the window's turn,
        however long the frame gap; otherwise yaw_rate_dps x dt. No yaw at all holds and logs.
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
    """No gated light, or an unknown color, votes go."""
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, GO)

    def update(
            self,
            detections: list[DetectionObject],
            log: list[str],
        ) -> str:
        """Logs lights under the confidence gate."""
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
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, False)

    def update(
            self,
            detections: list[DetectionObject],
            log: list[str],
        ) -> bool:
        """Logs signs under the confidence gate."""
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
    While the vote says a line is there, a missed frame repeats the last distances (px and
    cm from one frame); both are None when it says there isn't. tilt is never held.
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
        """(voted, distance_px, distance_cm). Logs a held distance."""
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
    """Held and voted as StopLineClassifier; None / False when the vote says there's none."""
    def __init__(self, cfg: Phase3Config) -> None:
        self._cfg = cfg
        self._vote = _Vote(cfg.vote_window, False)
        self._last: tuple[float | None, float | None, bool] = (None, None, False)

    def update(
            self,
            results: list[ObstacleResult],
            log: list[str],
        ) -> tuple[bool, float | None, float | None, bool]:
        """(voted, distance_px, distance_cm, clipped). Logs a held measurement."""
        seen = next((r for r in results if r.detected), None)
        if seen is not None:
            self._last = (seen.distance_px, seen.distance_cm, seen.clipped)
        if not self._vote.update(seen is not None):
            return False, None, None, False
        if seen is None:
            log.append(f"[OBSTACLE] missed this frame; holding {self._last[0]:.1f}px")
        return True, *self._last


class Phase3Processor:
    """Create once; call process() on every frame in order."""
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
        """Clamped to [0, max_dt_s]."""
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
        """(packet, debug summary). sensors=None means no sensors. ValueError if phase2 is None."""
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